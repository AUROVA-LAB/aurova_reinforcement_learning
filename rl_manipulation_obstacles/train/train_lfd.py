import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from data import *
from networks_lfd import *
from train_utils import collate_fn

import matplotlib.pyplot as plt
import cv2 as cv




# =========================================================
# TRAINING
# =========================================================


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HDF5LfDDataset(os.path.join(os.getcwd(), "../dataset"))

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn = collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle = True, collate_fn = collate_fn)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle = True, collate_fn = collate_fn)

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

    for epoch in range(50):

        # ================= TRAIN =================
        model.train()
        train_loss = 0

        for b in train_loader:
            print("Train")
            
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}

            pred = model(
                b["cam_D"],
                b["cam_ext_D"],
                b["gripper_pose"],
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

                pred = model(
                    b["cam_D"].to(device, non_blocking = True),
                    b["cam_ext_D"].to(device, non_blocking = True),
                    b["gripper_pose"].to(device, non_blocking = True),
                )

                val_loss = criterion(pred, b["action"]).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    # ================= TEST =================
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_loss = 0
    mae = 0

    with torch.no_grad():
        for b in test_loader:
            
            cam, cam_ext = b["cam"].to(device), b["cam_ext"].to(device)
            cam_D, cam_ext_D = b["cam_D"].to(device), b["cam_ext_D"].to(device)

            tgt_pose, gripper_pose = b["target_pose"].to(device), b["gripper_pose"].to(device)
            teacher_action = b["action"].to(device)

            pred = model(cam_D, cam_ext_D, gripper_pose)

            test_loss += criterion(pred, teacher_action).item()
            mae += torch.abs(pred - teacher_action).mean().item()

    test_loss /= len(test_loader)
    mae /= len(test_loader)

    print(f"\nTest MSE: {test_loss:.4f}")
    print(f"Test MAE: {mae:.4f}")


if __name__ == "__main__":
    train()