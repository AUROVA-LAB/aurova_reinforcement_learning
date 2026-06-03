import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from data import *
from networks_lfd import *
from train_utils import collate_fn, preprocess_img_sam, preprocess_img_sam2, preprocess_pcd

import matplotlib.pyplot as plt
import cv2 as cv


import time
# =========================================================
# TRAINING
# =========================================================


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HDF5LfDDataset(os.path.join(os.getcwd(), "../../dataset"))

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
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
        dataset = preprocess_pcd(dataset)

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn = collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle = True, collate_fn = collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle = True, collate_fn = collate_fn)

    # Get dimensions
    sample = dataset[0]
    pose_dim = sample["gripper_pose"].shape[0]
    action_dim = sample["action"].shape[0]


    in_channels = dataset[0]["cam_D"].shape[0]


    
    model = CnnPolicy(pose_dim, action_dim, 
                      in_channels = in_channels).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    best_val = float("inf")




    print("\n ------ Start training \n")


        


    for epoch in range(100):

        # ================= TRAIN =================
        model.train()
        train_loss = 0

        for b in train_loader:    
              
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}

            # f1 = b["cam_p"]
            # f2 = b["cam_ext_p"]
            # f3 = b["cam_front_p"]

            # pred = model(
            #     f1, f2, f3,
            #     b["gripper_pose"],
            # )
            pred = model(
                b["pcd_p"],
                b["sym"]
            )

            loss = criterion(pred, b["action"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for b in val_loader:
                
            
                b = {
                    k: v.to(device, non_blocking=True)
                    for k, v in b.items()
                }
            
                # f1 = b["cam_p"]
                # f2 = b["cam_ext_p"]
                # f3 = b["cam_front_p"]

                # pred = model(
                #     f1, f2, f3,
                #     b["gripper_pose"],
                # )
                pred = model(
                    b["pcd_p"],
                    b["sym"]
                )

                val_loss += criterion(pred, b["action"]).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pth")


if __name__ == "__main__":
    train()