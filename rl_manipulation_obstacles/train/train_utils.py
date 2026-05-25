import gymnasium as gym
import numpy as np
import torch

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab.utils.math import matrix_from_quat

import cv2 as cv

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator




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

    f = backbone.image_encoder(tensor).mean(dim = 1).view(f1.size(0), -1)
   

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

