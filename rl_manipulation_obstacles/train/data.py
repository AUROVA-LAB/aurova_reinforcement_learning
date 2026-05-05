import h5py
import numpy as np
import os

import torch
from torch.utils.data import Dataset



class HDF5EpisodeWriter:
    def __init__(self, output_dir, episode_idx, max_steps=1000):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.filepath = os.path.join(output_dir, f"episode_{episode_idx:03d}.h5")
        self.max_steps = max_steps
        self.step = 0

        self.file = h5py.File(self.filepath, "w")

        # Lazy init (we don’t know shapes yet)
        self.initialized = False

    def _init_datasets(self, cam_shape, cam_ext_shape, pose_dim, action_dim, gripper_action_dim):
        T = self.max_steps

        self.cam_ds = self.file.create_dataset(
            "images/cam", (T, *cam_shape), dtype="uint8"
        )
        self.cam_ext_ds = self.file.create_dataset(
            "images/cam_ext", (T, *cam_ext_shape), dtype="uint8"
        )

        self.target_pose_ds = self.file.create_dataset(
            "states/target_pose", (T, pose_dim), dtype="float32"
        )
        self.gripper_pose_ds = self.file.create_dataset(
            "states/gripper_pose", (T, pose_dim), dtype="float32"
        )

        self.action_ds = self.file.create_dataset(
            "actions", (T, action_dim), dtype="float32"
        )

        self.gripper_action_ds = self.file.create_dataset(
            "gripper_action", (T, 1), dtype="bool"
        )

        self.initialized = True

    def add_step(self, cam, cam_ext, target_pose, gripper_pose, action, gripper_action):
        """
        cam, cam_ext: (H, W, 3) uint8
        poses: (D,)
        action: (A,)
        """

        if not self.initialized:
            self._init_datasets(
                cam.shape,
                cam_ext.shape,
                target_pose.shape[0],
                action.shape[0],
                gripper_action
            )

        idx = self.step

        self.cam_ds[idx] = cam
        self.cam_ext_ds[idx] = cam_ext
        self.target_pose_ds[idx] = target_pose
        self.gripper_pose_ds[idx] = gripper_pose
        self.action_ds[idx] = action
        self.gripper_action_ds[idx] = gripper_action

        self.step += 1

    def close(self):
        # Trim unused space
        for key in self.file.keys():
            pass  # optional: skip resizing for simplicity

        self.file.close()





class HDF5LfDDataset(Dataset):
    def __init__(self, dataset_dir):
        self.files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".h5")]

        self.index = []  # (file_id, step)

        # Build global index
        for file_id, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                length = f["actions"].shape[0]
                for t in range(length):
                    self.index.append((file_id, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_id, t = self.index[idx]
        path = self.files[file_id]

        with h5py.File(path, "r") as f:
            cam = f["images/cam"][t]
            cam_ext = f["images/cam_ext"][t]

            target_pose = f["states/target_pose"][t]
            gripper_pose = f["states/gripper_pose"][t]
            action = f["actions"][t]

        # Convert to tensors
        cam = torch.tensor(cam).permute(2, 0, 1).float() / 255.0
        cam_ext = torch.tensor(cam_ext).permute(2, 0, 1).float() / 255.0

        target_pose = torch.tensor(target_pose).float()
        gripper_pose = torch.tensor(gripper_pose).float()
        action = torch.tensor(action).float()

        return {
            "cam": cam,
            "cam_ext": cam_ext,
            "target_pose": target_pose,
            "gripper_pose": gripper_pose,
            "action": action
        }