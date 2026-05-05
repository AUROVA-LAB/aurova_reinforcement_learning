import os
import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader, random_split

from data import *
from networks_lfd import *


# =========================================================
# TRAINING
# =========================================================

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HDF5LfDDataset("/workspace/dataset")

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    # Get dimensions
    sample = dataset[0]
    pose_dim = sample[2].shape[0]
    action_dim = sample[3].shape[0]

    model = CnnPolicy(pose_dim, action_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(50):

        # ================= TRAIN =================
        model.train()
        train_loss = 0

        for cam, cam_ext, pose, action in train_loader:
            cam, cam_ext = cam.to(device), cam_ext.to(device)
            pose, action = pose.to(device), action.to(device)

            pred = model(cam, cam_ext, pose)
            loss = criterion(pred, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for cam, cam_ext, pose, action in val_loader:
                cam, cam_ext = cam.to(device), cam_ext.to(device)
                pose, action = pose.to(device), action.to(device)

                pred = model(cam, cam_ext, pose)
                val_loss += criterion(pred, action).item()

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
        for cam, cam_ext, pose, action in test_loader:
            cam, cam_ext = cam.to(device), cam_ext.to(device)
            pose, action = pose.to(device), action.to(device)

            pred = model(cam, cam_ext, pose)

            test_loss += criterion(pred, action).item()
            mae += torch.abs(pred - action).mean().item()

    test_loss /= len(test_loader)
    mae /= len(test_loader)

    print(f"\nTest MSE: {test_loss:.4f}")
    print(f"Test MAE: {mae:.4f}")


if __name__ == "__main__":
    train()