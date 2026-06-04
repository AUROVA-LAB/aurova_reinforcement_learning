import gymnasium as gym
import numpy as np
import torch

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from isaaclab.utils.math import matrix_from_quat

import cv2 as cv

from segment_anything import sam_model_registry


from sam2.build_sam import build_sam2

from Pointnet_Pointnet2_pytorch.models.pointnet2_sem_seg import *
import open3d as o3d



def collate_fn(batch):

    out = {}

    for k in batch[0].keys():

        arr = np.stack([b[k] for b in batch])

        out[k] = torch.from_numpy(arr).float()

    return out



# --- Class for overriding and adding noise to the observations in the environment ---
class AddNoiseObservation(gym.ObservationWrapper):
    def __init__(self, env, noise_std = 0.1):
        super(AddNoiseObservation, self).__init__(env)
        self.noise_std = noise_std      # Standard deviation for noise adding

    # Add Gaussian noise to the observation --> Overrides method of GYM
    def observation(self, observation):
        '''
        In:
            - observation - dict: dictionary of observations. The key "policy" contains a tensor, ...
                ... which corresponds to the observation
                
        Out:
            - dict: dictionary where the key is "policy" and contains the noised observation.
        '''

        return {"policy": torch.normal(mean = observation["policy"], std = self.noise_std)}


# define models (stochastic and deterministic models) using mixins
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.action_layer(x)), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}
    

def depth_to_pointcloud(depth, fx, fy, cx, cy):
    """
    depth: (H, W)
    returns: (N, 3) points in camera frame
    """

    device = depth.device

    H, W = depth.shape

    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    z = depth
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    points = torch.stack((x, y, z), dim=-1)

    return points.reshape(-1, 3)



def transform_points(points, translation, quaternion):
    """
    points: (N, 3)
    translation: (3,)
    quaternion: (4,)  [x,y,z,w]
    """

    R = matrix_from_quat(quaternion)

    points_world = (R @ points.T).T + translation

    return points_world

def preprocess_img(img, backbone):
    '''
    In:
        - img: list of images
        - backbone: SAM image encoder
    '''

    img = np.transpose(img, (1, 2, 0))

    resized = cv.resize(img, (1024, 1024))

    tensor = torch.from_numpy(resized).float().cuda()

    # HWC -> BCHW
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    # normalize to SAM expected range
    # tensor = tensor / 255.0

    # SAM normalization
    pixel_mean = torch.tensor(
        [123.675, 116.28, 103.53],
        device=tensor.device
    ).view(1, 3, 1, 1)

    pixel_std = torch.tensor(
        [58.395, 57.12, 57.375],
        device=tensor.device
    ).view(1, 3, 1, 1)

    tensor = tensor * 255.0

    tensor = (tensor - pixel_mean) / pixel_std

    f = backbone.image_encoder(tensor).mean(dim = 1).view(tensor.size(0), -1)
   

    return f

def preprocess_img_sam(dataset, SAM_CHECKPOINT, SAM_TYPE):

    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)

    sam.to("cuda")
    sam.eval()

    backbone = sam

    for i in range(len(dataset)):
        print("--- Image ", i / len(dataset))
        print("------ Wrist")
        f1 = preprocess_img(dataset[i]["cam_D"], backbone)
        print("------ External")
        f2 = preprocess_img(dataset[i]["cam_ext_D"], backbone)
        print("------ Frontal")
        f3 = preprocess_img(dataset[i]["cam_front_D"], backbone)

        dataset.set_item(i, cam_p = f1,
                            cam_ext_p = f2,
                            cam_front_p = f3)


def preprocess_img_sam2(dataset):

    checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2 = build_sam2(model_cfg, checkpoint)
    
    backbone = sam2.image_encoder

    transform = T.Compose([
            T.ToPILImage(),
            T.Resize((1024, 1024)),   # depends on model config
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    

    print("preprocess for sam2")


    for i in range(len(dataset)):
        print("--- Image ", i / len(dataset))

        f1 = np.transpose(dataset[i]["cam_D"], (1,2,0))
        f2 = np.transpose(dataset[i]["cam_ext_D"], (1,2,0))
        f3 = np.transpose(dataset[i]["cam_front_D"], (1,2,0))

        f1 = transform(f1).unsqueeze(0).to("cuda:0")
        f2 = transform(f2).unsqueeze(0).to("cuda:0")
        f3 = transform(f3).unsqueeze(0).to("cuda:0")

        f = torch.cat((f1, f2, f3), dim = 0)

        f = backbone(f)['backbone_fpn'][-1].mean(dim = 1).view(f.shape[0], -1)

        dataset.set_item(i, cam_p = f[0],
                            cam_ext_p = f[1],
                            cam_front_p = f[2])
        
    return dataset


def preprocess_pcd(dataset):
    num_classes = 13
    model = get_model(num_classes=num_classes).cuda()

    checkpoint = torch.load(
        "best_model_sem.pth",
        map_location="cuda:0",
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()



    for i in range(len(dataset)):
        print("--- Image ", i / len(dataset))

        pc = dataset[i]["pc"].astype(np.float32)
        pc_ext = dataset[i]["pc_ext"].astype(np.float32)
        pc_front = dataset[i]["pc_front"].astype(np.float32)

        pc_all = np.concatenate([pc, pc_ext, pc_front], axis=0)

        # ============================================================
        # 2. VOXEL DOWNSAMPLE
        # ============================================================

        voxel_size = 0.025

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_all[:, :3])

        pcd = pcd.voxel_down_sample(voxel_size)

        pc_all = np.asarray(pcd.points)

        # keep RGB if exists
        if pc.shape[1] > 3:
            colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(pc_all)
        else:
            colors = np.zeros_like(pc_all)

        pc_all = np.concatenate([pc_all, colors], axis=1)


        # ============================================================
        # 3. FILTERING (your original logic cleaned)
        # ============================================================

        pc_all = pc_all[pc_all[:, 2] > 0.025]
        pc_all = pc_all[pc_all[:, 0] > -0.8]
        pc_all = pc_all[pc_all[:, 0] < 0.1]
        pc_all = pc_all[pc_all[:, 1] > -0.5]

        # ============================================================
        # 4. OPEN3D POINT CLOUD + NORMALS
        # ============================================================

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pc_all[:, :3])


        # ============================================================
        # 5. POINTNET FEATURES (conv1 output)
        # ============================================================

        pc_xyz = pc_all[:, :3]
        pc_rgb = pc_all[:, 3:] if pc_all.shape[1] > 3 else np.zeros_like(pc_xyz)

        # normalize
        pc_xyz = pc_xyz - np.mean(pc_xyz, axis=0)
        # pc_xyz = pc_xyz / (np.max(np.linalg.norm(pc_xyz, axis=1)) + 1e-8)


        # sample for PointNet
        NUM_POINTS = 4096
        if len(pc_xyz) == 0.0:
            continue
        idx = np.random.choice(len(pc_xyz), NUM_POINTS, replace=len(pc_xyz) < NUM_POINTS)

        xyz_sample = pc_xyz[idx]
        rgb_sample = pc_rgb[idx]


        # ============================================================
        # 6. BUILD POINTNET INPUT
        # ============================================================

        xyz_norm = xyz_sample / (np.max(np.abs(xyz_sample), axis=0, keepdims=True) + 1e-8)

        features_in = np.concatenate(
            [
                xyz_sample,      # xyz
                xyz_norm,        # normalized xyz
                rgb_sample       # optional color
            ],
            axis=1
        )

        points = torch.tensor(features_in.T, dtype=torch.float32).unsqueeze(0).cuda()



        with torch.no_grad():
            _, _, point_features = model(points)

        point_features = point_features.mean(-1)[0].cpu().numpy()

        dataset.set_item(i, pcd_p = point_features)

    return dataset
        


def farthest_point_sampling(points, n_samples):
    """
    points: (N, D) numpy array
    n_samples: number of points to sample

    Returns:
        sampled_points: (n_samples, D)
        sampled_indices: (n_samples,)
    """
    N = points.shape[0]

    sampled_indices = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(N, np.inf)

    # Randomly choose the first point
    farthest = np.random.randint(0, N)

    for i in range(n_samples):
        sampled_indices[i] = farthest

        centroid = points[farthest]

        # Compute squared distances to the newly selected point
        dist = np.sum((points - centroid) ** 2, axis=1)

        # Keep minimum distance to any selected point
        distances = np.minimum(distances, dist)

        # Next farthest point
        farthest = np.argmax(distances)

    return points[sampled_indices], sampled_indices



def preprocess_pcd_raw(dataset):

    for i in range(len(dataset)):
        print("--- Image ", i / len(dataset))

        pc = dataset[i]["pc"].astype(np.float32)
        pc_ext = dataset[i]["pc_ext"].astype(np.float32)
        pc_front = dataset[i]["pc_front"].astype(np.float32)

        pc_all = np.concatenate([pc[:, :3], pc_ext[:, :3], pc_front[:, :3]], axis=0)

        # ============================================================
        # 2. VOXEL DOWNSAMPLE
        # ============================================================

        voxel_size = 0.025

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_all[:, :3])

        pcd = pcd.voxel_down_sample(voxel_size)

        pc_all = np.asarray(pcd.points)

        # ============================================================
        # 3. FILTERING (your original logic cleaned)
        # ============================================================

        pc_all = pc_all[pc_all[:, 2] > 0.025]
        pc_all = pc_all[pc_all[:, 0] > -0.8]
        pc_all = pc_all[pc_all[:, 0] < 0.1]
        pc_all = pc_all[pc_all[:, 1] > -0.5]

        # pts = torch.from_numpy(pc_all).float()[None]  # (1, N, 3)

        if pc_all.shape[0] != 0:

            sampled_pts, sampled_idx = farthest_point_sampling(
                pc_all,
                n_samples=512
            )

            # rigid perturbation
            t = np.random.normal(0, 0.005, (3,))
            R = o3d.geometry.get_rotation_matrix_from_xyz(
                np.random.normal(0, np.deg2rad(2), (3,))
            )

            sampled_pts = sampled_pts @ R.T + t

            # jitter
            sampled_pts += np.clip(
                np.random.normal(0, 0.003, sampled_pts.shape),
                -0.01,
                0.01
            )

            # dropout
            mask = np.random.rand(len(sampled_pts)) > 0.05
            sampled_pts = sampled_pts[mask]

            # restore fixed size
            sampled_pts, _ = farthest_point_sampling(sampled_pts, 512)

            # cloud = o3d.geometry.PointCloud()
            # cloud.points = o3d.utility.Vector3dVector(sampled_pts)

            # o3d.visualization.draw_geometries([cloud])

            dataset.set_item(i, pcd_p = sampled_pts)

    return dataset