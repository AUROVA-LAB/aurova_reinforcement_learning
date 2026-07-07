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
    test_loader = DataLoader(test_ds, batch_size=32, shuffle = False, collate_fn = collate_fn)

    # Get dimensions
    sample = dataset[0]
    pose_dim = sample["gripper_pose"].shape[0]
    action_dim = sample["action"].shape[0]


    # in_channels = dataset[0]["cam_D"].shape[0]


    
    model = CnnPolicy(pose_dim, action_dim, 
                      in_channels = 3,
                      pc=True,
                      hidden_dim=64).to(device)
    criterion = nn.SmoothL1Loss()
    criterion2 = nn.MSELoss()
    
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    sl1_loss = 0
    mse_loss = 0
    mae_loss = 0

    dataset.max_action = 1.0
    dataset.max_gripper = 1.0

    with open("action_preprocessing.pkl","rb") as f:
        stats = pickle.load(f)

    dataset.max_pc = stats["max_pc"]
    dataset.min_pc = stats["min_pc"]

    with torch.no_grad():
        for b in test_loader:
            
            b = {
                    k: v.to(device, non_blocking=True)
                    for k, v in b.items()
                }

            pc =  b["pc_net3_seq"].to(device)
            pose = b["pose_seq"].to(device)
            traj = b["diff"].to(device)

            pred = model(pc)

            sample_mag=torch.norm(
                traj,
                dim=1
            )

            sample_mag=(
                sample_mag/
                sample_mag.mean()
            )

            print(pc.max())
            print(pc.min())
            print(traj.max())
            print(traj.min())
            print(traj)
            print(sample_mag)
            print("-------")

            # traj = torch.tensor(qt.inverse_transform(traj.cpu().numpy()))
            # pred = torch.tensor(qt.inverse_transform(pred.cpu().numpy()))

            

            # print(pc)
            # print(pose)
            # print(traj)
            # print(pred)

            print("SL1: ", criterion(pred, traj))
            print("MSE: ", criterion2(pred, traj))
            print("L1: ", torch.abs(pred - traj).mean(dim=0))

            sl1_loss += criterion(pred, traj)
            mse_loss += criterion2(pred, traj)
            mae_loss += torch.abs(pred - traj).mean().item()
            print("----")

    sl1_loss /= len(test_loader)
    mse_loss /= len(test_loader)
    mae_loss /= len(test_loader)

    print(f"\nTest MSE: {mse_loss:.4f}")
    print(f"Test MAE: {mae_loss:.4f}")
    print(f"Test Smooth MAE: {sl1_loss:.4f}")


if __name__ == "__main__":
    test()