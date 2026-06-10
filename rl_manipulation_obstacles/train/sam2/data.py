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

    def _init_datasets(self, cam_shape = None, cam_ext_shape = None, cam_front_shape = None, 
                       cam_p_shape = None, cam_ext_p_shape = None, cam_front_p_shape = None, 
                       pcd_p_shape = None, pcd_net_shape = None, 
                       pc_shape = None, pc_ext_shape = None, pc_front_shape = None, 
                       pose_dim = None, action_dim = None, gripper_action_dim = None):
        T = self.max_steps

        self.cam_ds = self.file.create_dataset(
            "images/cam", (T, *cam_shape), dtype="float32"
        )
        self.cam_ext_ds = self.file.create_dataset(
            "images/cam_ext", (T, *cam_ext_shape), dtype="float32"
        )
        self.cam_front_ds = self.file.create_dataset(
            "images/cam_front", (T, *cam_front_shape), dtype="float32"
        )

        
        
        self.cam_p_ds = self.file.create_dataset(
            "images/cam_p", (T, *cam_p_shape), dtype="float32"
        )
        self.cam_ext_p_ds = self.file.create_dataset(
            "images/cam_ext_p", (T, *cam_ext_p_shape), dtype="float32"
        )
        self.cam_front_p_ds = self.file.create_dataset(
            "images/cam_front_p", (T, *cam_front_p_shape), dtype="float32"
        )

        if pcd_p_shape is not None:
            self.pcd_p_ds = self.file.create_dataset(
                "pc/pcd_p", (T, *pcd_p_shape), dtype="float32"
            )

        if pcd_net_shape is not None:
            self.pcd_net_ds = self.file.create_dataset(
                "pc/pcd_net", (T, *pcd_net_shape), dtype="float32"
            )


        if pc_shape is not None:
            self.pc_ds = self.file.create_dataset(
                "pc/pc", (T, *pc_shape), dtype="float32"
            )

        if pc_ext_shape is not None:
            self.pc_ext_ds = self.file.create_dataset(
                "pc/pc_ext", (T, *pc_ext_shape), dtype="float32"
            )

        if pc_front_shape is not None:
            self.pc_front_ds = self.file.create_dataset(
                "pc/pc_front", (T, *pc_front_shape), dtype="float32"
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
            "gripper_action", (T, 1), dtype="float32"
        )

        self.initialized = True

    def add_step(self, cam, cam_ext, cam_front, 
                 cam_p, cam_ext_p, cam_front_p,
                 pcd_p, pcd_net,
                 pc_w, pc_ext, pc_front, 
                 target_pose, gripper_pose, action, diff, gripper_action):
        """
        cam, cam_ext: (H, W, 3) uint8
        pc_w, pc_ext, pc_front: (N, 3) float32
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
                cam_p.shape,
                cam_ext_p.shape,
                cam_front_p.shape,
                pcd_p.shape,
                pcd_net.shape,
                pc_w.shape,
                pc_ext.shape,
                pc_front.shape,
                target_pose.shape[0],
                action.shape[0],
                gripper_action
            )

        idx = self.step

        self.cam_ds[idx] = cam
        self.cam_ext_ds[idx] = cam_ext
        self.cam_front_ds[idx] = cam_front
        self.cam_p_ds[idx] = cam_p
        self.cam_ext_p_ds[idx] = cam_ext_p
        self.cam_front_p_ds[idx] = cam_front_p
        self.pcd_p_ds[idx] = pcd_p
        self.pcd_net_ds[idx] = pcd_net
        self.pc_ds[idx] = pc_w
        self.pc_ext_ds[idx] = pc_ext
        self.pc_front_ds[idx] = pc_front
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





# =========================================================
# BASE HDF5 DATASET
# =========================================================

class HDF5LfDDataset(Dataset):

    def __init__(
        self,
        dataset_dir,
        obs_horizon=5,
        pred_horizon=5,
        stride=1
    ):
        super().__init__()

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.stride = stride
        
        self.p_scale = 10
        self.H = 5
        self.mean_diff = 0.0
        self.std_diff = 1.0
        self.mean_sym = 0.0
        self.std_sym = 1.0

        # -------------------------------------------------
        # Load files
        # -------------------------------------------------
        self.files = sorted([
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith(".h5")
        ])

        self.handles = [
            h5py.File(path, "r+")
            for path in self.files
        ]
        # -------------------------------------------------
        # Build index of valid windows
        # -------------------------------------------------
        self.index = []

        for file_id, f in enumerate(self.handles):

            if "actions" not in f:
                continue

            T = f["actions"].shape[0]

            max_start = T - (obs_horizon + pred_horizon)

            for t in range(0, max_start, stride):
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

        file_id, t0 = self.index[idx]
        f = self.handles[file_id]

        # cam = f["/images/cam"][t0]
        # cam_ext = f["/images/cam_ext"][t0]
        # cam_front = f["/images/cam_front"][t0]

        # cam_p = f["/images/cam_p"][t0]
        # cam_ext_p = f["/images/cam_ext_p"][t0]
        # cam_front_p = f["/images/cam_front_p"][t0]

        pc = f["/pc/pc"][t0]
        pc_ext = f["/pc/pc_ext"][t0]
        pc_front = f["/pc/pc_front"][t0]

        pcd_p = f["/pc/pcd_p"][t0]

        target_pose = f["/states/target_pose"][t0]
        gripper_pose = f["/states/gripper_pose"][t0] 

        action = f["actions"][t0] 
        # prev_action = f["actions"][t - 1]  if t > 0 else np.zeros_like(action)
        diff = f["diff"][t0] 
        gripper_action = f["gripper_action"][t0]




        T_obs = self.obs_horizon
        T_pred = self.pred_horizon

        t1 = t0 + T_obs
        t2 = t1 + T_pred

        # -------------------------------------------------
        # OBSERVATIONS (SEQUENCE)
        # -------------------------------------------------

        pc_seq = f["/pc/pcd_p"][t0:t1]            # [T_obs, 512, 3]
        pc_net_seq = f["/pc/pcd_net"][t0:t1]            # [T_obs, 512, 3]
        pose_seq = f["/states/gripper_pose"][t0:t1]
        sym_seq = (f["/states/target_pose"][t0:t1]
                   - f["/states/gripper_pose"][t0:t1])
        

        # -------------------------------------------------
        # ACTION TRAJECTORY (TARGET)
        # -------------------------------------------------

        traj = f["actions"][t1:t2]                # [T_pred, 6]
        diff_seq = f["diff"][t1:t2]

        # optional: gripper
        # if "gripper_action" in f:
        #     gripper = f["gripper_action"][t1:t2]
        #     traj = np.concatenate([traj, gripper], axis=-1)


        
        return {
            # "cam": cam[:-1],
            # "cam_D": np.repeat(cam[-1][None], 3, axis=0),
            # "cam_p": cam_p,

            # "cam_ext": cam_ext[:-1],
            # "cam_ext_D": np.repeat(cam_ext[-1][None], 3, axis=0),
            # "cam_ext_p": cam_ext_p,

            # "cam_front": cam_front[:-1],
            # "cam_front_D": np.repeat(cam_front[-1][None], 3, axis=0),
            # "cam_front_p": cam_front_p,
            
            "pc": pc,
            "pc_ext": pc_ext,
            "pc_front": pc_front,

            "pcd_p": pcd_p,

            # "target_pose": target_pose,
            "gripper_pose": gripper_pose,
            "action": action, #np.concatenate([action, gripper_action], axis=-1),
            "diff": diff,
            # "prev_action": prev_action
            "sym": (target_pose - gripper_pose - self.mean_sym) / (self.std_sym + 1e-8),

            # --- Traj ---
            # Observations
            "pc_seq": torch.tensor(pc_seq, dtype=torch.float32),
            "pc_net_seq": torch.tensor(pc_net_seq, dtype=torch.float32),
            "pose_seq": torch.tensor(pose_seq, dtype=torch.float32),
            "sym_seq": torch.tensor(sym_seq, dtype=torch.float32),
            # Actions
            "traj": torch.tensor(traj, dtype=torch.float32),
            "diff_seq": torch.tensor(diff_seq, dtype=torch.float32)
        }

    # =====================================================
    # MODIFY ELEMENTS
    # =====================================================

    def set_item(
        self,
        idx,
        cam=None,
        cam_ext=None,
        cam_front=None,
        cam_p=None,
        cam_ext_p=None,
        cam_front_p=None,
        pcd_p = None,
        pcd_net = None,
        pc_raw = None,
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
            elif cam.ndim == 3 and cam.shape[-1] == 1:
                f["/images/cam"][t][-1] = cam

            # float -> uint8
            if cam.dtype != np.uint8:
                cam = (cam * 255).astype(np.uint8)

            # f["/images/cam"][t] = cam

        if cam_ext is not None:

            if torch.is_tensor(cam_ext):
                cam_ext = cam_ext.cpu().numpy()

            # HWC -> CWH
            if cam_ext.ndim == 3 and cam_ext.shape[-1] == 3:
                cam_ext = np.transpose(cam_ext, (2, 0, 1))
            elif cam_ext.ndim == 3 and cam_ext.shape[-1] == 1:
                f["/images/cam_ext"][t][-1] = cam_ext

            if cam_ext.dtype != np.uint8:
                cam_ext = (cam_ext * 255).astype(np.uint8)

            # f["cam_ext"][t] = cam_ext

        if cam_front is not None:

            if torch.is_tensor(cam_front):
                cam_front = cam_front.cpu().numpy()

            # HWC -> CWH
            if cam_front.ndim == 3 and cam_front.shape[-1] == 3:
                cam_front = np.transpose(cam_front, (2, 0, 1))
            elif cam_front.ndim == 3 and cam_front.shape[-1] == 1:
                f["/images/cam_front"][t][-1] = cam_front

            if cam_front.dtype != np.uint8:
                cam_front = (cam_front * 255).astype(np.uint8)

            # f["cam_front"][t] = cam_front

        
        if cam_p is not None:
            if torch.is_tensor(cam_p):
                cam_p = cam_p.detach().cpu().numpy()

            if cam_p.ndim == 1 and cam_p.shape[0] == 64*64:
                f["/images/cam_p"][t] = cam_p*self.p_scale
                                

            # # float -> uint8
            # if cam_p.dtype != np.uint8:
            #     cam_p = (cam_p * 255).astype(np.uint8)

        if cam_ext_p is not None:

            if torch.is_tensor(cam_ext_p):
                cam_ext_p = cam_ext_p.detach().cpu().numpy()

            # HWC -> CWH
            if cam_ext_p.ndim == 1 and cam_ext_p.shape[0] == 64*64:
                f["/images/cam_ext_p"][t] = cam_ext_p*self.p_scale
                

            # # float -> uint8
            # if cam_ext_p.dtype != np.uint8:
            #     cam_ext_p = (cam_ext_p * 255).astype(np.uint8)

        if cam_front_p is not None:

            if torch.is_tensor(cam_front_p):
                cam_front_p = cam_front_p.detach().cpu().numpy()

            # HWC -> CWH
            if cam_front_p.ndim == 1 and cam_front_p.shape[0] == 64*64:
                f["/images/cam_front_p"][t] = cam_front_p*self.p_scale
                

            # # float -> uint8
            # if cam_front_p.dtype != np.uint8:
            #     cam_front_p = (cam_front_p * 255).astype(np.uint8)


        if pcd_p is not None:
            if torch.is_tensor(pcd_p):
                pcd_p = pcd_p.detach().cpu().numpy()

            if pcd_p.ndim == 2 and pcd_p.shape[0] == 512:
                f["/pc/pcd_p"][t] = pcd_p
                print("Setting ...")


        if pcd_net is not None:
            if torch.is_tensor(pcd_net):
                pcd_net = pcd_net.detach().cpu().numpy()

            if pcd_net.ndim == 1 and pcd_net.shape[0] == 128:
                f["/pc/pcd_net"][t] = pcd_net
                print("Setting NET...")
        


        f.flush()


