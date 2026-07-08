import gymnasium as gym
import numpy as np
import torch

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt


import cv2 as cv

# from segment_anything import sam_model_registry


# from sam2.build_sam import build_sam2

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))



from Pointnet_Pointnet2_pytorch.models.pointnet2_sem_seg import *
import open3d as o3d

from networks_lfd import FastDCTFeatureReducer
from Point_BERT.models.Point_BERT import PointTransformer
from easydict import EasyDict
from sklearn.preprocessing import QuantileTransformer, RobustScaler, MinMaxScaler
import pickle





torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False



def collate_fn(batch):

    out = {}

    for k in batch[0].keys():

        arr = np.stack([b[k] for b in batch])

        out[k] = torch.from_numpy(arr).float()

    return out


@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))



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



def normalize_pc(pc):
        centroid = pc.mean(dim=1, keepdim=True)

        pc_centered = pc - centroid
        scale = (
            torch.linalg.norm(pc_centered, dim=1)
            .max(dim=0, keepdim=True)
            .values
            .unsqueeze(-1)
        )  # (B,1,1)

        pc_norm = pc_centered / scale.squeeze(-1)

        return pc_norm, centroid, scale

def preprocess_pcd_single(pc_all, model, mode="BERT"):

    # ============================================================
    # 2. VOXEL DOWNSAMPLE
    # ============================================================

    voxel_size = 0.015

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_all[:, :3])

    pcd = pcd.voxel_down_sample(voxel_size)

    pc_all = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # pc_all = np.concatenate([pc_all, colors], axis=1)


    # ============================================================
    # 3. FILTERING (your original logic cleaned)
    # ============================================================

    pc_all = pc_all[pc_all[:, 2] > 0.025]
    pc_all = pc_all[pc_all[:, 0] > -0.8]
    pc_all = pc_all[pc_all[:, 0] < 0.1]
    pc_all = pc_all[pc_all[:, 1] > -0.5]

    # Add noise

    # pc_all, _ = farthest_point_sampling(pc_all, 512)
    noised = add_noise_to_pcd(points = pc_all[:, :3])[0]
    pc_all[:, :3] = noised
    # ============================================================
    # 4. OPEN3D POINT CLOUD + NORMALS
    # ============================================================

    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(pc_all[:, :3])


    # ============================================================
    # 5. POINTNET FEATURES (conv1 output)
    # ============================================================

    pc_xyz = pc_all[:, :3]
    # pc_rgb = pc_all[:, 3:]#  if pc_all.shape[1] > 3 else np.zeros_like(pc_xyz)

    pc_xyz, _, _ = normalize_pc(torch.tensor(pc_xyz).unsqueeze(0))


    if mode == "PointNet2":
        sampled_pts, sampled_idx = farthest_point_sampling(
                pc_xyz,
                1024
            )  
        xyz_sample = sampled_pts
        rgb_sample = pc_rgb[sampled_idx]


    elif mode == "BERT":
        sampled_pts = farthest_point_sampling_BERT(
                pc_xyz,
                1024
            ) 
        
        xyz_sample = sampled_pts

    

    # ============================================================
    # 6. BUILD POINTNET INPUT
    # ============================================================

    if mode == "BERT":
        points = torch.tensor(
            xyz_sample,
            dtype=torch.float32
        ).cuda()

        with torch.no_grad():
            # forward_features exists in PointTransformer
            __, point_features, __  = model(points)
            point_features = point_features.squeeze()


    elif mode == "PointNet2":
        # xyz_norm = xyz_sample / (np.max(np.abs(xyz_sample), axis=0, keepdims=True) + 1e-8)
        # xyz_norm, _, _ = normalize_pc(torch.tensor(xyz_sample))

        features_in = np.concatenate(
            [
                xyz_sample,      # xyz
                xyz_sample,        # normalized xyz
                rgb_sample       # optional color
            ],
            axis=1
        )

        points = torch.tensor(features_in.T, dtype=torch.float32).unsqueeze(0).to("cuda:0")

        with torch.no_grad():
            _, _, point_features = model(points)
            point_features = point_features[0].permute(1,0)
            # point_features = dct_reducer.encode(point_features[0]).permute(1,0)
            # point_features = (point_features - torch.mean(point_features)) / torch.std(point_features)


    # point_features = point_features.cpu().numpy()
    return point_features


def preprocess_pcd(dataset, mode = "BERT", test_curr_max = None, test = False):

    curr_max = 0.0

    if mode == "PointNet2":
        num_classes = 13
        model = get_model(num_classes=num_classes).cuda()

        torch.cuda.init()

        checkpoint = torch.load(
            "best_model_sem.pth",
            map_location="cuda:0",
            weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to("cuda:0")
        model.eval()
    elif mode == "BERT":
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

        model = PointTransformer(config)

        # load pretrained checkpoint
        # ckpt = torch.load(
        #     "Point-BERT.pth",
        #     map_location="cpu",
        #     weights_only=False
        # )

        # # depending on checkpoint structure:
        # model.load_state_dict(
        #     ckpt['base_model'],
        #     strict=False
        # )

        model.load_model_from_ckpt(
            bert_ckpt_path="Point-BERT.pth",)

        model.eval()
        model.cuda()

    actions_list = []
    pos_list = []

    if test_curr_max is None:
        for i in range(len(dataset)):
            print("Calculating ", i, " max")
            pc = dataset[i]["pc"].astype(np.float32)
            pc_ext = dataset[i]["pc_ext"].astype(np.float32)
            pc_front = dataset[i]["pc_front"].astype(np.float32)
            pc_all = torch.Tensor(np.concatenate([pc, pc_ext, pc_front], axis=0))
            new_max_pc = torch.max(torch.abs(pc_all)).item()

            if new_max_pc > curr_max:
                curr_max = new_max_pc

            actions_list.append(dataset[i]["diff"])
            pos_list.append(dataset[i]["gripper_pose"])

        actions_list = np.array(actions_list)
        pos_list = np.array(pos_list)


        qt = RobustScaler()
        actions_norm = qt.fit_transform(actions_list)

        qt_pos = RobustScaler()
        pos_norm = qt_pos.fit_transform(pos_list)

        actions_minmax = MinMaxScaler(feature_range=(-1,1))
        actions_norm = actions_minmax.fit_transform(actions_norm)

        pos_minmax = MinMaxScaler(feature_range=(-1,1))
        pos_norm = pos_minmax.fit_transform(pos_norm)

        for i in range(len(dataset)):
            dataset.set_item(i, diff = actions_norm[i], gripper_pose = pos_norm[i])

    else:

        for i in range(len(dataset)):
            print("Calculating ", i, " max")

            actions_list.append(dataset[i]["diff"])
            pos_list.append(dataset[i]["gripper_pose"])

        actions_list = np.array(actions_list)
        pos_list = np.array(pos_list)

        with open("action_preprocessing.pkl","rb") as f:
            stats = pickle.load(f)

        curr_max = test_curr_max

        qt = stats["qt_pc"]
        actions_norm = qt.transform(actions_list)

        
        qt_pos = stats["qt_pos"]
        pos_norm = qt_pos.transform(pos_list)

        actions_minmax = stats["actions_minmax"]
        pos_minmax = stats["pos_minmax"]

        actions_norm = actions_minmax.transform(actions_norm)
        
        pos_norm = pos_minmax.transform(pos_norm)

        for i in range(len(dataset)):
            dataset.set_item(i, diff = actions_norm[i], gripper_pose = pos_norm[i])


        pc_mean = stats["pc_mean"]
        pc_std = stats["pc_std"]
        max_pc = stats["max_pc"]
        min_pc = stats["min_pc"]
        



    
    pc_data = []

    for i in range(len(dataset)):
        print("--- Image ", i / len(dataset))

        # Preprocess PCs
        pc = dataset[i]["pc"].astype(np.float32)  / curr_max
        pc_ext = dataset[i]["pc_ext"].astype(np.float32)  / curr_max
        pc_front = dataset[i]["pc_front"].astype(np.float32)  / curr_max

        pc_all = np.concatenate([pc, pc_ext, pc_front], axis=0)

        point_features = preprocess_pcd_single(pc_all, model, mode = mode)

        
        if point_features is None:
            continue

        point_features = point_features.cpu().numpy()

        pc_data.append(point_features)

        if mode == "PointNet2":
            dataset.set_item(i, pcd_net2 = point_features)
        elif mode == "BERT":
            dataset.set_item(i, pcd_net3 = point_features)




    pc_data = np.array(pc_data)

    if not test:
        pc_mean = np.mean(pc_data, axis = 0)
        pc_std = np.std(pc_data, axis = 0)

    # pc_data = (pc_data - pc_mean)/(pc_std + 1e-8)
    
    if not test:
        max_pc = np.max(point_features, axis = -1)
        min_pc = np.min(point_features, axis = -1)
        
        stats = {
            "qt_pc": qt,
            "qt_pos": qt_pos,
            "pc_mean": pc_mean,
            "pc_std": pc_std,
            "max_pc": max_pc,
            "min_pc": min_pc,
            "actions_minmax": actions_minmax,
            "pos_minmax": pos_minmax,
        }

        with open("action_preprocessing.pkl","wb") as f:
            pickle.dump(stats,f)

        # with open("action_preprocessing.pkl","rb") as f:
        #     stats = pickle.load(f)

    dataset.max_pc = max_pc
    dataset.min_pc = min_pc

    return dataset, curr_max
        



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
        dist = np.sum((points[:, :3] - centroid[:3]) ** 2, axis=1)

        # Keep minimum distance to any selected point
        distances = np.minimum(distances, dist)

        # Next farthest point
        farthest = np.argmax(distances)

    return points[sampled_indices], sampled_indices

def farthest_point_sampling_BERT(xyz, npoint):
    xyz = torch.tensor(xyz).float()#.unsqueeze(0)
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long)
    distance = torch.ones(B, N) * 1e10
    farthest = torch.zeros(B, dtype=torch.long)

    batch_indices = torch.arange(B, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest

        centroid = xyz[batch_indices, farthest].view(B,1,3)

        dist = ((xyz - centroid)**2).sum(-1)

        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = distance.max(-1)[1]

    return xyz[:, centroids.squeeze()].numpy()



def add_noise_to_pcd(points, 
                     t_std = [0, 0.005],
                     angle_std = [0, np.deg2rad(2)],
                     jitter_std = [0, 0.003], jitter_clip = 0.01,
                     dropout_rate = 0.05):
    
    N = points.shape[0]

    # rigid perturbation
    t = np.random.normal(t_std[0], t_std[1], (3,))
    R = o3d.geometry.get_rotation_matrix_from_xyz(
        np.random.normal(angle_std[0], angle_std[1], (3,))
    )

    points = points @ R.T + t

    # jitter
    points += np.clip(
        np.random.normal(jitter_std[0], jitter_std[1], points.shape),
        -jitter_clip,
        jitter_clip
    )

    # dropout
    mask = np.random.rand(len(points)) > dropout_rate
    points = points[mask]

    # restore fixed size
    points = farthest_point_sampling_BERT(torch.tensor(points).unsqueeze(0), N)

    return points


def preprocess_single_pcd_raw(pc_all, mean_dataset):
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


        # Add noise and center point cloud around gripper
        sampled_pts = add_noise_to_pcd(sampled_pts)#  - mean_dataset[3:]*2

        # cloud = o3d.geometry.PointCloud()
        # cloud.points = o3d.utility.Vector3dVector(sampled_pts)

        # o3d.visualization.draw_geometries([cloud])

    return sampled_pts


def preprocess_pcd_raw(dataset):

    max_pc = 0.0
    max_gripper = 0.0
    max_action = 0.0

    for i in range(len(dataset)):
        print("--- Image ", i / len(dataset))

        pc = dataset[i]["pc"].astype(np.float32)
        pc_ext = dataset[i]["pc_ext"].astype(np.float32)
        pc_front = dataset[i]["pc_front"].astype(np.float32)

        pc_all = np.concatenate([pc[:, :3], pc_ext[:, :3], pc_front[:, :3]], axis=0)

        sampled_pts = preprocess_single_pcd_raw(pc_all, dataset[i]["gripper_pose"])

        dataset.set_item(i, pcd_p = sampled_pts)


        # Preprocess gemometrical info
        norm_gripper = (dataset[i]["gripper_pose"])# - np.mean(dataset[i]["gripper_pose"])) / np.std(dataset[i]["gripper_pose"])
        norm_action = (dataset[i]["action"])# - np.mean(dataset[i]["action"])) / np.std(dataset[i]["action"])

        new_max_gripper = np.max(np.abs(norm_gripper))
        new_max_action = np.max(np.abs(norm_action))
        new_max_pc = np.max(np.abs(sampled_pts))

        if new_max_pc > max_pc:
            max_pc = new_max_pc
        if new_max_gripper > max_gripper:
            max_gripper = new_max_gripper
        if new_max_action > max_action:
            max_action = new_max_action

    dataset.max_pc = max_pc
    dataset.max_gripper = max_gripper
    dataset.max_action = max_action


    # mean_diff /= len(dataset)
    # std_diff = np.std(mean_diff, axis=0)

    # dataset.mean_diff = mean_diff
    # dataset.std_diff = std_diff

    # mean_sym /= len(dataset)
    # std_sym = np.std(mean_sym, axis=0)

    # dataset.mean_sym = mean_sym
    # dataset.std_sym = std_sym

    return dataset