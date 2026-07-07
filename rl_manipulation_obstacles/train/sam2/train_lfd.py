import os
import torch
import torch.nn as nn
import numpy as np

from data import *
from networks_lfd import *
from train_utils import *

import matplotlib.pyplot as plt
import cv2 as cv

import random
import wandb
from datetime import datetime

import time
# =========================================================
# TRAINING
# =========================================================
wandb.init(
    project="lfd-robotics",
    name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    config={
        "lr": 5e-4,
        "optimizer": "AdamW",
        "loss": "L1Loss",
        "hidden_dim": 64,
        "model": "CnnPolicy"
    }
)

import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    random_split,
    WeightedRandomSampler
)

# =========================================================
# TRAINING
# =========================================================

def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    dataset = HDF5LfDDataset(
        os.path.join(os.getcwd(), "../../dataset")
    )

    MODE="pcd"

    if MODE=="pcd":
        dataset,curr_max=preprocess_pcd(dataset)

    train_size=int(0.8*len(dataset))
    val_size=int(0.1*len(dataset))
    test_size=len(dataset)-train_size-val_size

    train_ds,val_ds,test_ds=random_split(
        dataset,
        [train_size,val_size,test_size]
    )

    ##################################################
    # COMPUTE SAMPLE WEIGHTS
    ##################################################

    ##################################################
    # COMPUTE SAMPLE WEIGHTS FROM FREQUENCY
    ##################################################

    train_actions = []

    for idx in train_ds.indices:

        a = dataset[idx]["diff"]
        train_actions.append(a)

    train_actions = np.array(train_actions)

    ##################################################
    # Compute action magnitude
    ##################################################

    magnitudes = np.linalg.norm(
        train_actions,
        axis=1
    )

    ##################################################
    # Build histogram
    ##################################################

    num_bins = 20

    hist, bin_edges = np.histogram(
        magnitudes,
        bins=num_bins
    )

    ##################################################
    # Assign each sample to a bin
    ##################################################

    bin_ids = np.digitize(
        magnitudes,
        bin_edges[:-1]
    )

    bin_ids = np.clip(
        bin_ids,
        0,
        num_bins-1
    )

    ##################################################
    # Inverse frequency weighting
    ##################################################

    sample_weights = np.zeros(
        len(magnitudes)
    )

    for i in range(len(magnitudes)):

        freq = hist[bin_ids[i]]

        sample_weights[i] = (
            1.0/(freq+1e-6)
        )

    ##################################################
    # Normalize
    ##################################################

    sample_weights /= sample_weights.mean()

    sample_weights = torch.DoubleTensor(
        sample_weights
    )

    ##################################################
    # Weighted sampler
    ##################################################

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    ##################################################
    # DATALOADERS
    ##################################################

    train_loader=DataLoader(
        train_ds,
        batch_size=32,
        sampler=sampler,
        collate_fn=collate_fn,
        shuffle=False,
    )

    val_loader=DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader=DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    sample=dataset[0]

    pose_dim=sample["gripper_pose"].shape[0]
    action_dim=sample["diff"].shape[0]

    model=CnnPolicy(
        pose_dim,
        action_dim,
        in_channels=3,
        hidden_dim=64,
        pc=True
    ).to(device)

    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2
    )

    best_val=float("inf")

    global_step=0

    print("\n------ Start training ------\n")

    config = EasyDict({
            'trans_dim':384,
            'depth':12,
            'drop_path_rate':0.1,
            'cls_dim':40,
            'num_heads':6,
            'group_size':32,
            'num_group':128,
            'encoder_dims':256
        })

    backbone = PointTransformer(config)

    backbone.load_model_from_ckpt(
        bert_ckpt_path="Point-BERT.pth",)

    backbone.eval()
    backbone.cuda()

    for epoch in range(500):

        ########################################
        # TRAIN
        ########################################

        model.train()

        train_loss=0

        for b in train_loader:

            b={
                k:v.to(
                    device,
                    non_blocking=True
                )
                for k,v in b.items()
            }

            pcds = b["pc_all_seq"]

            B, T, N, _ = pcds.shape
            pcds = pcds.view(B*T, N, -1)

            p_f = torch.zeros((B*T, 768))

            for l in range(B*T):
                pc = (pcds[l] / curr_max).cpu().numpy()
                features = preprocess_pcd_single(pc, mode="BERT", model = backbone)

                p_f[l] = torch.tensor(features).detach().clone()
            
            p_f = 2*(p_f - dataset.min_pc) / (dataset.max_pc - dataset.min_pc) - 1 
            p_f = p_f.view(B,T,768)
            p_f = torch.tensor(p_f).detach().clone().to(device)

            pc= p_f # b["pc_net3_seq"]
            traj=b["diff"]

            pred=model(pc)

            ##################################
            # MAGNITUDE-WEIGHTED LOSS
            ##################################

            sample_mag=torch.norm(
                traj,
                dim=1
            )

            sample_mag=(
                sample_mag/
                sample_mag.mean()
            )

            loss_per_sample=F.smooth_l1_loss(
                pred,
                traj,
                reduction='none'
            ).mean(dim=1)

            loss=(
                sample_mag*
                loss_per_sample
            ).mean()

            ##################################

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss+=loss.item()

            wandb.log({
                "epoch":epoch,
                "train/batch_loss":loss.item(),
                "sample_mag_mean":
                    sample_mag.mean().item()
            },
            step=global_step)

            global_step+=1

        train_loss/=len(train_loader)

        ########################################
        # VALIDATION
        ########################################

        model.eval()

        val_loss=0

        mae_per_dim=[]

        with torch.no_grad():

            for b in val_loader:

                b={
                    k:v.to(device)
                    for k,v in b.items()
                }

                pcds = b["pc_all_seq"]

                B, T, N, _ = pcds.shape
                pcds = pcds.view(B*T, N, -1)

                p_f = torch.zeros((B*T, 768))

                for l in range(B*T):
                    pc = (pcds[l] / curr_max).cpu().numpy()
                    features = preprocess_pcd_single(pc, mode="BERT", model = backbone)

                    p_f[l] = torch.tensor(features).detach().clone()
                
                p_f = 2*(p_f - dataset.min_pc) / (dataset.max_pc - dataset.min_pc) - 1 
                p_f = p_f.view(B,T,768)
                p_f = torch.tensor(p_f).detach().clone().to(device)

                pc= p_f # b["pc_net3_seq"]
                traj=b["diff"]

                pred=model(pc)

                loss=F.smooth_l1_loss(
                    pred,
                    traj
                )

                val_loss+=loss.item()

                mae=torch.abs(
                    pred-traj
                ).mean(0)

                mae_per_dim.append(
                    mae.cpu().numpy()
                )

        val_loss/=len(val_loader)

        mae_per_dim=np.mean(
            mae_per_dim,
            axis=0
        )

        wandb.log({

            "train/epoch_loss":
                train_loss,

            "val/epoch_loss":
                val_loss,

            **{
                f"mae_dim_{i}":v
                for i,v
                in enumerate(mae_per_dim)
            }
        })

        print(
            f"Epoch {epoch}"
            f" Train {train_loss:.4f}"
            f" Val {val_loss:.4f}"
        )

        print(
            "Per-dim MAE:",
            mae_per_dim
        )

        if val_loss<best_val:

            best_val=val_loss

            torch.save(
                model.state_dict(),
                "best_model.pth"
            )

            print(
                "SAVING BEST MODEL"
            )


    ##################################################
    # TEST
    ##################################################

    print("\n------ TESTING BEST MODEL ------\n")

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=device
        )
    )

    model.eval()

    test_loss = 0
    test_mse = 0
    test_mae = 0

    mae_per_dim = []

    pred_mag_all = []
    target_mag_all = []

    with torch.no_grad():

        for b in test_loader:

            b = {
                k:v.to(device)
                for k,v in b.items()
            }

            pcds = b["pc_all_seq"]

            B, T, N, _ = pcds.shape
            pcds = pcds.view(B*T, N, -1)

            p_f = torch.zeros((B*T, 768))

            for l in range(B*T):
                pc = (pcds[l] / curr_max).cpu().numpy()
                features = preprocess_pcd_single(pc, mode="BERT", model = backbone)

                p_f[l] = torch.tensor(features).detach().clone()
            
            p_f = 2*(p_f - dataset.min_pc) / (dataset.max_pc - dataset.min_pc) - 1 
            p_f = p_f.view(B,T,768)
            p_f = torch.tensor(p_f).detach().clone().to(device)

            pc= p_f # b["pc_net3_seq"]
            traj = b["diff"]

            pred = model(pc)

            #################################
            # LOSSES
            #################################

            smooth = F.smooth_l1_loss(
                pred,
                traj
            )

            mse = F.mse_loss(
                pred,
                traj
            )

            mae = F.l1_loss(
                pred,
                traj
            )

            test_loss += smooth.item()
            test_mse += mse.item()
            test_mae += mae.item()

            #################################
            # PER-DIMENSION MAE
            #################################

            mae_dim = torch.abs(
                pred-traj
            ).mean(dim=0)

            mae_per_dim.append(
                mae_dim.cpu().numpy()
            )

            #################################
            # MAGNITUDES
            #################################

            pred_mag = torch.norm(
                pred,
                dim=1
            )

            target_mag = torch.norm(
                traj,
                dim=1
            )

            pred_mag_all.extend(
                pred_mag.cpu().numpy()
            )

            target_mag_all.extend(
                target_mag.cpu().numpy()
            )


    #################################
    # FINAL STATISTICS
    #################################

    test_loss /= len(test_loader)
    test_mse /= len(test_loader)
    test_mae /= len(test_loader)

    mae_per_dim = np.mean(
        mae_per_dim,
        axis=0
    )

    pred_mag_mean = np.mean(
        pred_mag_all
    )

    target_mag_mean = np.mean(
        target_mag_all
    )

    print("\n========== TEST RESULTS ==========")

    print(
        f"SmoothL1: {test_loss:.6f}"
    )

    print(
        f"MSE: {test_mse:.6f}"
    )

    print(
        f"L1: {test_mae:.6f}"
    )

    print(
        "\nPer-dim MAE:"
    )

    for i,m in enumerate(mae_per_dim):

        print(
            f"Action {i}: {m:.6f}"
        )

    print(
        f"\nPrediction magnitude mean: {pred_mag_mean:.6f}"
    )

    print(
        f"Target magnitude mean: {target_mag_mean:.6f}"
    )

    #################################
    # WANDB
    #################################

    wandb.log({

        "test/smoothl1": test_loss,
        "test/mse": test_mse,
        "test/l1": test_mae,

        "test/pred_mag_mean":
            pred_mag_mean,

        "test/target_mag_mean":
            target_mag_mean,

        **{
            f"test/mae_dim_{i}":v
            for i,v
            in enumerate(mae_per_dim)
        }
    })


if __name__ == "__main__":
    train()