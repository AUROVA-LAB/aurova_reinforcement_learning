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

    def _init_datasets(self, cam_shape, cam_ext_shape, cam_front_shape, pose_dim, action_dim, gripper_action_dim):
        T = self.max_steps

        self.cam_ds = self.file.create_dataset(
            "images/cam", (T, *cam_shape), dtype="uint8"
        )
        self.cam_ext_ds = self.file.create_dataset(
            "images/cam_ext", (T, *cam_ext_shape), dtype="uint8"
        )
        self.cam_front_ds = self.file.create_dataset(
            "images/cam_front", (T, *cam_front_shape), dtype="uint8"
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

        self.diff_ds = self.file.create_dataset(
            "diff", (T, action_dim), dtype="float32"
        )

        self.gripper_action_ds = self.file.create_dataset(
            "gripper_action", (T, 1), dtype="bool"
        )

        self.initialized = True

    def add_step(self, cam, cam_ext, cam_front, target_pose, gripper_pose, action, diff, gripper_action):
        """
        cam, cam_ext: (H, W, 3) uint8
        poses: (D,)
        action: (A,)
        diff: (A,)
        gripper_action: bool
        """

        if not self.initialized:
            self._init_datasets(
                cam.shape,
                cam_ext.shape,
                cam_front.shape,
                target_pose.shape[0],
                action.shape[0],
                gripper_action
            )

        idx = self.step

        self.cam_ds[idx] = cam
        self.cam_ext_ds[idx] = cam_ext
        self.cam_front_ds[idx] = cam_front
        self.target_pose_ds[idx] = target_pose
        self.gripper_pose_ds[idx] = gripper_pose
        self.action_ds[idx] = action
        self.diff_ds[idx] = diff
        self.gripper_action_ds[idx] = gripper_action

        self.step += 1

    def close(self):
        # # Trim unused space
        # for key in self.file.keys():
        #     pass  # optional: skip resizing for simplicity

        # self.file.close()
        self.file.flush()
        self.file.close()



import os
import h5py
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms


# =========================================================
# BASE HDF5 DATASET
# =========================================================

class HDF5LfDDataset(Dataset):

    def __init__(self, dataset_dir):

        self.files = sorted([
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith(".h5")
        ])

        # -------------------------------------------------
        # Persistent HDF5 handles
        # -------------------------------------------------

        self.handles = [
            h5py.File(path, "r")
            for path in self.files
        ]

        # -------------------------------------------------
        # Global indexing
        # -------------------------------------------------

        self.index = []  # (file_id, timestep)

        for file_id, f in enumerate(self.handles):
            print(list(f.keys()))
            length = f["actions"].shape[0]

            for t in range(length):

                if np.all(f["actions"][t] == 0.0):
                    continue

                self.index.append((file_id, t))

    # =====================================================
    # CLEANUP
    # =====================================================

    def close(self):

        for h in self.handles:
            h.close()

    def __del__(self):
        self.close()

    # =====================================================
    # LENGTH
    # =====================================================

    def __len__(self):
        return len(self.index)

    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(self, idx):

        file_id, t = self.index[idx]
        f = self.handles[file_id]

        cam = f["/images/cam"][t]
        cam_ext = f["/images/cam_ext"][t]
        cam_front = f["/images/cam_front"][t]

        target_pose = f["/states/target_pose"][t]
        gripper_pose = f["/states/gripper_pose"][t] 

        action = f["actions"][t] 
        prev_action = f["actions"][t - 1]  if t > 0 else np.zeros_like(action)
        diff = f["diff"][t] 
        gripper_action = f["gripper_action"][t]

        
        return {
            "cam": cam[:-1],
            "cam_D": cam[-1][None] / 255.0,
            "cam_ext": cam_ext[:-1],
            "cam_ext_D": cam_ext[-1][None] / 255.0,
            "cam_front": cam_front[:-1],
            "cam_front_D": cam_front[-1][None] / 255.0,
            "target_pose": target_pose,
            "gripper_pose": gripper_pose,
            "action": np.concatenate([action, gripper_action], axis=-1),
            "diff": np.concatenate([action, gripper_action], axis=-1),
            "prev_action": prev_action
        }

    # =====================================================
    # MODIFY ELEMENTS
    # =====================================================

    def set_item(
        self,
        idx,
        cam=None,
        cam_ext=None,
        target_pose=None,
        gripper_pose=None,
        action=None,
        gripper_action=None
    ):

        file_id, t = self.index[idx]

        f = self.handles[file_id]

        # -------------------------------------------------
        # CAM
        # -------------------------------------------------

        if cam is not None:

            if torch.is_tensor(cam):
                cam = cam.cpu().numpy()

            # HWC -> CWH
            if cam.ndim == 3 and cam.shape[-1] == 3:
                cam = np.transpose(cam, (2, 0, 1))

            # float -> uint8
            if cam.dtype != np.uint8:
                cam = (cam * 255).astype(np.uint8)

            f["cam"][t] = cam

        # -------------------------------------------------
        # CAM EXT
        # -------------------------------------------------

        if cam_ext is not None:

            if torch.is_tensor(cam_ext):
                cam_ext = cam_ext.cpu().numpy()

            # HWC -> CWH
            if cam_ext.ndim == 3 and cam_ext.shape[-1] == 3:
                cam_ext = np.transpose(cam_ext, (2, 0, 1))

            if cam_ext.dtype != np.uint8:
                cam_ext = (cam_ext * 255).astype(np.uint8)

            f["cam_ext"][t] = cam_ext

        # -------------------------------------------------
        # TARGET POSE
        # -------------------------------------------------

        if target_pose is not None:

            if torch.is_tensor(target_pose):
                target_pose = target_pose.cpu().numpy()

            f["target_pose"][t] = target_pose

        # -------------------------------------------------
        # GRIPPER POSE
        # -------------------------------------------------

        if gripper_pose is not None:

            if torch.is_tensor(gripper_pose):
                gripper_pose = gripper_pose.cpu().numpy()

            f["gripper_pose"][t] = gripper_pose

        # -------------------------------------------------
        # ACTION
        # -------------------------------------------------

        if action is not None:

            if torch.is_tensor(action):
                action = action.cpu().numpy()

            f["actions"][t] = action

        # -------------------------------------------------
        # GRIPPER ACTION
        # -------------------------------------------------

        if gripper_action is not None:

            if torch.is_tensor(gripper_action):
                gripper_action = gripper_action.cpu().numpy()

            f["gripper_action"][t] = gripper_action

        f.flush()




# =========================================================
# PROCESSED DATASET
# =========================================================

class ProcessedDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        image_size=(128, 128),
        use_relative_pose=False,
        use_delta_actions=False,
        normalize_images=False
    ):

        self.dataset = base_dataset

        self.use_relative_pose = use_relative_pose
        self.use_delta_actions = use_delta_actions

        # -------------------------------------------------
        # IMAGE TRANSFORMS
        # -------------------------------------------------

        transform_list = [
            transforms.Resize(image_size)
        ]

        if normalize_images:

            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        self.transform = transforms.Compose(transform_list)

    # =====================================================
    # LENGTH
    # =====================================================

    def __len__(self):
        return len(self.dataset)

    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        cam = sample["cam"]
        cam_ext = sample["cam_ext"]

        target_pose = sample["target_pose"]
        gripper_pose = sample["gripper_pose"]

        action = sample["action"]
        gripper_action = sample["gripper_action"]

        # -------------------------------------------------
        # IMAGE PROCESSING
        # -------------------------------------------------

        cam = self.transform(cam)
        cam_ext = self.transform(cam_ext)

        # -------------------------------------------------
        # RELATIVE POSE
        # -------------------------------------------------

        if self.use_relative_pose:

            pose = target_pose - gripper_pose

        else:

            pose = torch.cat([
                target_pose,
                gripper_pose
            ], dim=-1)

        # -------------------------------------------------
        # DELTA ACTIONS
        # -------------------------------------------------

        if self.use_delta_actions:

            if idx < len(self.dataset) - 1:

                next_sample = self.dataset[idx + 1]

                next_action = next_sample["action"]

                action = next_action - action

            else:

                action = torch.zeros_like(action)

        return {
            "cam": cam,
            "cam_ext": cam_ext,
            "pose": pose,
            "action": action,
            "gripper_action": gripper_action
        }