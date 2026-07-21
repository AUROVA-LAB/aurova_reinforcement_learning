import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from data import *
from networks_lfd import *
from train_utils import *

import matplotlib.pyplot as plt
import cv2 as cv




def test():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HDF5LfDDataset(os.path.join(os.getcwd(), "../../dataset"))

    # Split dataset
    train_size = int(0.05 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size



    MODE = "pcd"

    YOLO_MODEL = "yolov8n.pt"

    SAM_CHECKPOINT = "./sam_vit_h_4b8939.pth"
    SAM_TYPE = "vit_h"


    if MODE == "yolo":
        model = YOLO(YOLO_MODEL)

        backbone = model.model.model#.model          # DetectMultiBackend

        # Extract backbone (layers 0–9)
        backbone = nn.Sequential(*backbone[:10], 
                                nn.AdaptiveAvgPool2d((1,1)), 
                                nn.Flatten(), 
                                nn.Linear(512, 128))
        backbone.eval()

    
    elif MODE == "sam":
        dataset = preprocess_img_sam(dataset, SAM_CHECKPOINT, SAM_TYPE)

    elif MODE == "resnet":
        pass

    elif MODE == "sam2":
        dataset = preprocess_img_sam2(dataset)

    elif MODE == "pcd":
        dataset, curr_max = preprocess_pcd(dataset, test_curr_max=1.11878, test = True)



    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn = collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle = False, collate_fn = collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle = False, collate_fn = collate_fn)

    # Get dimensions
    sample = dataset[0]
    pose_dim = sample["gripper_pose"].shape[0]
    action_dim = sample["action"].shape[0]


    # in_channels = dataset[0]["cam_D"].shape[0]


    
    model = CnnPolicy(pose_dim, action_dim*3, 
                      in_channels = 3,
                      pc=True,
                      hidden_dim=64).to(device)
    # criterion = nn.SmoothL1Loss()
    # criterion2 = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()
    
    model.load_state_dict(torch.load("best_model_BERT_cat2.pth"))
    model.eval()

    sl1_loss = 0
    mse_loss = 0
    mae_loss = 0
    test_loss = 0
    test_cat = 0
    test_mag = 0

    dataset.max_action = 1.0
    dataset.max_gripper = 1.0

    with open("action_preprocessing_BERT_cat2.pkl","rb") as f:
        stats = pickle.load(f)

    dataset.max_pc = stats["max_pc"]
    dataset.min_pc = stats["min_pc"]

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

    criterion = nn.BCEWithLogitsLoss()
    criterion_mag = nn.MSELoss()


    with torch.no_grad():
        for b in test_loader:
            
            b = {
                    k: v.to(device, non_blocking=True)
                    for k, v in b.items()
                }

            # pcds = b["pc_all_seq"][:,:,:,:3]

            # B, T, N, a = pcds.shape
            # pcds = pcds.view(B*T, N, -1)

            # p_f = torch.zeros((B*T, 768))

            # pc = pcds / curr_max
            # p_f = preprocess_pcd_single_batch(pc, mode="BERT", model = backbone)
            
            # p_f = 2*(p_f - dataset.min_pc) / (dataset.max_pc - dataset.min_pc) - 1 
            # p_f = p_f.view(B,T,768)
            # p_f = torch.tensor(p_f).detach().clone().to(device)

            pc= b["pc_net3_seq"] # p_f
            traj=b["cat_diff"]
            traj_mag = b["mag"]

            pred, pred_mag = model(pc)

            #################################
            # LOSSES
            #################################

            # smooth = F.smooth_l1_loss(
            #     pred,
            #     traj
            # )

            # mse = F.mse_loss(
            #     pred,
            #     traj
            # )

            # mae = F.l1_loss(
            #     pred,
            #     traj
            # )

            

            loss_cat = criterion(pred, traj)
            loss_mag = criterion_mag(pred_mag, traj_mag)
            loss = loss_cat + loss_mag



            test_loss += loss.item()
            test_cat += loss_cat.item()
            test_mag += loss_mag.item()
            # test_loss += smooth.item()
            # test_mse += mse.item()
            # test_mae += mae.item()

            #################################
            # PER-DIMENSION MAE
            #################################

            # mae_dim = torch.abs(
            #     pred-traj
            # ).mean(dim=0)

            # mae_per_dim.append(
            #     mae_dim.cpu().numpy()
            # )

            #################################
            # MAGNITUDES
            #################################

            # pred_mag = torch.norm(
            #     pred,
            #     dim=1
            # )

            # target_mag = torch.norm(
            #     traj,
            #     dim=1
            # )

            # pred_mag_all.extend(
            #     pred_mag.cpu().numpy()
            # )

            # target_mag_all.extend(
            #     target_mag.cpu().numpy()
            # )


    #################################
    # FINAL STATISTICS
    #################################

    test_loss /= len(test_loader)
    test_cat /= len(test_loader)
    test_mag /= len(test_loader)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"\nTest Cat: {test_cat:.4f}")
    print(f"\nTest Mag: {test_mag:.4f}")
    # print(f"Test MAE: {mae_loss:.4f}")
    # print(f"Test Smooth MAE: {sl1_loss:.4f}")


if __name__ == "__main__":
    test()