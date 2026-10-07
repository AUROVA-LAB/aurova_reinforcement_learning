from py_dq.src.dq import *
from py_dq.src.distances import *
from py_dq.src.dq_lie import *
from py_dq.src.interpolators import *

from py_dq.src.quat_trans_lie import *
from py_dq.src.matrix_lie import *
from py_dq.src.euler import *

import numpy as np

# --- Mapping configuration ---
DQ = 0
EULER = 1
QUAT = 2
MAT = 3

# Size of the Lie algebra
sizes = [[8, 6, 7, 16], [6]*4]

representation = MAT
mapping = 1
size = sizes[int(mapping != 0)][representation]
size_group = sizes[0][representation]
distance = 0
symmetry = True

# Scalings for each action
scalings = [[[0.01, 0.001], [1.552, 0.822], [1, 0.574]],
            [[0.007, 0.02]],
            [[0.006, 0.025],],
            [[0.02,  0.004], [0.988,  0.602], [0.02, 0.004]]]

action_scaling = scalings[representation][mapping]




# --- Lie Algebra ---
# List of mappings
map_list = [[[identity_map, identity_map], [exp_bruno, log_bruno],     [exp_stereo, log_stereo]],
            [[identity_map, identity_map],],
            [[identity_map, identity_map], [exp_quat_stereo, log_quat_stereo]],
            [[identity_map, identity_map], [exp_se3, log_se3], [exp_gram_se3, log_gram_se3]]]

# List of conversions
conversions = [[convert_dq_to_Lab, dq_from_tr], 
                [convert_euler_to_Lab, from_quat_to_euler], 
                [convert_quat_trans_to_Lab, identity_map_conversion], 
                [convert_homo_to_Lab, homo_from_mat_trans_LAB]]

# Difference and multiply operators
diff_operators = [dq_diff, euler_diff, q_trans_diff, mat_diff]
mul_operators = [dq_mul, euler_mul, q_trans_mul, mat_mul]

# List of interpolators
interpolators = [ScLERP, None, None, None]

# Lis of distance functions
distances = [[dqLOAM_distance, geodesic_dist, double_geodesic_dist],
                [geodesic_dist],
                [geodesic_dist],
                [geodesic_dist]]

# Identities for each group
identities = [[1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                [0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0,   1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0,   0.0, 1.0, 0.0, 0.0,   0.0, 0.0, 1.0, 0.0,   0.0, 0.0, 0.0, 1.0]]

# Normaliaztion functions
normalizes = [dq_normalize, euler_normalize, norm_quat, norm_mat]


def convert_np_euler_2_map(pose, log, to_group):
    pose_torch = torch.tensor(pose).unsqueeze(0)
            
    quat_torch = quat_from_euler_xyz(pose_torch[:, 3], pose_torch[:, 4], pose_torch[:, 5])
    
    pose_lab = torch.cat((pose_torch[:, :3], quat_torch), dim = -1)
    pose_group = to_group(t = pose_lab[:, :3], r = pose_lab[:, 3:])

    return log(pose_group).squeeze(0).cpu().numpy(), pose_group.squeeze(0).cpu().numpy()
            


def sample_random_pose(
    pos_low=(-1.0, -1.0, -1.0),
    pos_high=(1.0, 1.0, 1.0),
    rot_low=(-np.pi, -np.pi, -np.pi),
    rot_high=(np.pi, np.pi, np.pi),
):
    """
    Sample a random 6D pose.

    Pose:
        [x, y, z, rx, ry, rz]

    Rotations are represented as Euler angles in radians.
    """
    position = np.random.uniform(pos_low, pos_high)
    rotation = np.random.uniform(rot_low, rot_high)

    return torch.tensor(np.concatenate([position, rotation]).astype(np.float32))


def pose_distance(pose, goal, log, to_group, rotation_weight, position_weight):
    """
    Euclidean distance between two 6D poses.

    Returns:
        position_distance
        rotation_distance
    """
    

    pose_map, pose_group = convert_np_euler_2_map(pose, log, to_group)
    goal_map, goal_group = convert_np_euler_2_map(goal, log, to_group)
   

    position_distance = np.linalg.norm(
        pose_map[3:] / position_weight - goal_map[3:] / position_weight
    )

    rotation_distance = np.linalg.norm(
        pose_map[:3] / rotation_weight - goal_map[:3] / rotation_weight
    )

    return position_distance, rotation_distance


def compute_reward(
    pose,
    goal,
    log,
    to_group,
    position_weight=1.0,
    rotation_weight=1.0,
):
    """
    Negative distance-to-goal reward.
    """
    

    position_distance, rotation_distance = pose_distance(
        pose, goal, log, to_group, rotation_weight, position_weight
    )

    reward = -(
        position_weight * position_distance
        + rotation_weight * rotation_distance
    )

    return reward
