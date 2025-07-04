from __future__ import annotations

import os
from math import pi
import torch
from collections.abc import Sequence
import numpy as np
from scipy.spatial.transform import Rotation

from .py_dq.src.lie import *

import copy

from omni.isaac.lab_tasks.manager_based.classic.aurova_reinforcement_learning.rl_manipulation.robots_cfg import UR5e_4f_CFG, UR5e_3f_CFG, GEN3_4f_CFG, UR5e_NOGRIP_CFG
from .mdp.utils import compute_rewards, save_images_grid

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import euler_xyz_from_quat, matrix_from_quat
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.assets import RigidObjectCfg

from omni.isaac.lab.managers import EventTermCfg, SceneEntityCfg
from omni.isaac.lab.envs import mdp

'''
                    ############## IMPORTANT #################
   The whole environment is build for two robots: the UR5e and Kinova GEN3-7dof.
   These two variables (cfg.UR5e and cfg.GEN3) serve as an abstraction to treat the robots during the episodes. In fact,
all the methods need an index to differentiate from which robot get the information.
   Also, data storage is performed using lists, not tensors because the joint space of the robots is
different from one another.
'''


# Function to change a Euler angles to a quaternion as a tensor
def rot2tensor(rot: Rotation) -> torch.tensor:
    '''
    In:
        - rot - scipy.Rotation(3): rotation expressed in Euler angles.

    Out:
        - rot_tensor_quat - torch.tensor - (4): rotation expressed a quaternion in a tensor.
    '''

    # Transform rotation to tensor
    rot_tensor = torch.tensor(rot.as_quat())
    rot_tensor_quat = torch.zeros((4))

    # Scipy uses the notation (x,y,z,w) whilst IsaacLab uses (w,x,y,z), so that is changed
    rot_tensor_quat[0], rot_tensor_quat[1:] = rot_tensor[-1].clone(), rot_tensor[:3].clone()
    
    return rot_tensor_quat

# Rotations respecto to the end effector robot link frame for object spawning
rot_45_z_pos = Rotation.from_rotvec(pi/4 * np.array([0, 0, 1]))        # Positive 45 degrees rotation in Z axis 
rot_180_z_pos = Rotation.from_rotvec(pi * np.array([0, 0, 1]))        # Positive 180 degrees rotation in Z axis 
rot_305_z_neg = Rotation.from_rotvec(-5*pi/4 * np.array([0, 0, 1]))     # Negative 135 degrees rotation in Z axis 
rot_45_z_pos = Rotation.from_rotvec((pi/4) * np.array([0, 0, 1]))
rot_90_x_pos = Rotation.from_rotvec(pi/2 * np.array([1, 0, 0]))         # Positive 90 degrees rotation in X axis


@configclass
class EventCfg:
  robot_physics_material = EventTermCfg(
      func=mdp.randomize_rigid_body_material,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg("UR5e_NOGRIP", body_names=".*"),
          "static_friction_range": (0.7, 1.3),
          "dynamic_friction_range": (0.7, 1.2),
          "restitution_range": (0.5, 1),
          "num_buckets": 250,
      },
  )
  robot_joint_stiffness_and_damping = EventTermCfg(
      func=mdp.randomize_actuator_gains,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg("UR5e_NOGRIP", joint_names=".*"),
          "stiffness_distribution_params": (0.75, 1.5),
          "damping_distribution_params": (0.75, 1.5),
          "operation": "scale",
          "distribution": "log_uniform",
      },
  )
#   reset_gravity = EventTermCfg(
#       func=mdp.randomize_physics_scene_gravity,
#       mode="interval",
#       is_global_time=True,
#       interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
#       params={
#           "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
#           "operation": "add",
#           "distribution": "gaussian",
#       },
#   )

#   body_mass = EventTermCfg(
#       func = mdp.randomize_rigid_body_mass,
#       mode = "reset",
#       params = {
#           "asset_cfg": SceneEntityCfg("object"),
#           "mass_distribution_params": (0.5, 1.5),
#           "operation": "scale",
#           "distribution": "uniform",
#           "recompute_inertia": False,
#       }
#   )

# Configuration class for the environment
@configclass
class RLManipulationDirectCfg(DirectRLEnvCfg):

    # events: EventCfg = EventCfg()
    
    # ---- Env variables ----
    decimation = 3              # Number of control action updates @ sim dt per policy dt.
    episode_length_s = 3.0      # Length of the episode in seconds
    max_steps = 320              # Maximum steps in an episode

    seq_len = 2                 # Length of the sequence
   
    option = 0                 # Option for the NN (0: everything, 1: pre-trained MLP, 2: pre-trained MLP with GNN)

    models = [["2025-05-06_13-10-55/model", "2025-05-06_18-49-55/model", "2025-05-08_18-22-52/model"],
              ["2025-05-07_01-15-07/model_46080000_steps"],
              ["2025-05-07_10-31-30/model", "2025-05-09_05-58-42/model", "2025-05-09_08-56-41/model"],
              []]

    # --- Mapping configuration ---
    DQ = 0
    EULER = 1
    QUAT = 2
    MAT = 3

    # Size of the Lie algebra
    sizes = [[8, 6, 7, 16], [6]*4]
    
    representation = DQ
    mapping = 0
    size = sizes[int(mapping != 0)][representation]
    size_group = sizes[0][representation]
    distance = 0
    path_to_pretrained = models[representation][mapping] # Path to the pre-trained approaching model

    scalings = [[[0.01, 0.001], [0.07,  0.003], [0.01, 0.007]],
                [[0.007, 0.02]],
                [[0.006, 0.025], [0.006, 0.03], [0.007, 0.015], [0.007, 0.015]],
                [[0.02,  0.004], [0.03,  0.006]]]
    grip_scaling = 5

    action_scaling = scalings[representation][mapping]



    # --- Action / observation space ---
    num_actions = size + 1   # Number of actions per environment (overridden)
    num_observations = size + 1 #* (seq_len)                         # Number of observations per environment (overridden)
    # state_space = 0
    


    num_envs = 1                # Number of environments by default (overriden)

    debug_markers = True        # Activate marker visualization
    save_imgs = False           # Activate image saving from cameras
    render_imgs = False          # Activate image rendering
    render_steps = 6            # Render images every certain amount of steps

    velocity_limit = 10         # Velocity limit for robots' end effector


    UR5e = 0                    # Robot options
    GEN3 = 1
    UR5e_3f = 2
    UR5e_NOGRIP = 3
    
    robot = UR5e_3f

    keys = ['UR5e', 'GEN3', 'UR5e_3f', 'UR5e_NOGRIP']     # Keys for the robots in simulation
    ee_link = ['tool0',         # Names for the end effector of each robot
               'tool_frame',
               'tool0',
               'wrist_3_link']
    
    grip_theta_max = [1.2218, 1.5708]
    m = [grip_theta_max[0]/140, grip_theta_max[1]/100]

    def_pos = [0.0, 0.0496, 0.0, -0.0523]

    open = [def_pos[1], 
            def_pos[0], def_pos[0],
            def_pos[2],
            def_pos[1], def_pos[1],
            def_pos[3],
            def_pos[2], def_pos[2],
            def_pos[3], def_pos[3]]
    close = copy.deepcopy(open)
    close[0] = 0.65
    close[4] = 0.65
    close[5] = 0.65

    close[-1] = -0.65
    close[-2] = -0.65
    close[-5] = -0.65

    moving_joints_gripper = [0.0, 
                             0.0, m[0], 
                             m[0],
                             m[0], 0.0,
                             0.0,
                             0.0, -m[0],
                             -m[0], -m[0]]
    

    # ---- Configurations ----
    # Simulation
    sim: SimulationCfg = SimulationCfg(dt = 1/max_steps, render_interval = decimation)
    # SimulationCfg: configuration for simulation physics 
    #    dt: time step of the simulation (seconds)
    #    render_interval: number of physics steps per rendering steps

    # Robots
    robot_cfg_1: Articulation = UR5e_4f_CFG.replace(prim_path="/World/envs/env_.*/" + keys[UR5e])
    robot_cfg_2: Articulation = GEN3_4f_CFG.replace(prim_path="/World/envs/env_.*/" + keys[GEN3])
    robot_cfg_3: Articulation = UR5e_3f_CFG.replace(prim_path="/World/envs/env_.*/" + keys[UR5e_3f])
    robot_cfg_4: Articulation = UR5e_NOGRIP_CFG.replace(prim_path="/World/envs/env_.*/" + keys[UR5e_NOGRIP])

    
    # Markers
    marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "ur5e_ee_pose": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
                visible = debug_markers
            ),
            "target_point": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
                visible = debug_markers
            ),
            "target_point2": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
                visible = debug_markers
            ),
        }
    )
    # VisualizationMarkersCfg: A class to configure a VisualizationMarkers.
    #    markers: The dictionary of marker configurations.
    #       UsdFileCfg: USD file to spawn asset from. --> In this case, a frame prim is imported from its USD file.

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs = num_envs, env_spacing = 2.5, replicate_physics = True)
    # InteractiveSceneCfg: Configuration for the interactive scene.
    #    num_envs: Number of environment instances handled by the scene.
    #    env_spacing: Spacing between environments. --> Positions are automatically handled
    #    replicate_physics: Enable/disable replication of physics schemas when using the Cloner APIs. If True, the simulation will have the same asset instances (USD prims) in all the cloned environments.
    
    # Object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cuboid",

        spawn=sim_utils.CuboidCfg(
            size=(0.045, 0.3, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity = False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.000025),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled = True,
                                                            contact_offset=0.0075),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos = [-1, -0.11711,  0.05]),
    )
    # RigidObjectCfg: Configuration parameters for a rigid object.
    #    spawn: Spawn configuration for the asset. --> Deciding which object type it is spawned
    #       CuboidCfg: Configuration parameters for a cuboid prim.
    #          size: Size of the cuboid.
    #          rigid_props / mass_props / collision_props / visual_material: properties of the prim declaration
    #    init_state: Initial state of the rigid object. --> Initial pose



    # ---- Joint information ----
    # Robot joint names
    joints = [['arm_shoulder_pan_joint', 'arm_shoulder_lift_joint', 'arm_elbow_joint', 'arm_wrist_1_joint', 'arm_wrist_2_joint', 'arm_wrist_3_joint'],
              ['arm_joint_1', 'arm_joint_2', 'arm_joint_3', 'arm_joint_4', 'arm_joint_5', 'arm_joint_6', 'arm_joint_7'],
              ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
              ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']]
    
    # Hand joint names
    hand_joints = [['joint_' + str(i) + '_0' for i in range(0,16)] for i in range(2)] + \
            [['robotiq_finger_middle_joint_1', 'robotiq_palm_finger_1_joint', 'robotiq_palm_finger_2_joint', 'robotiq_finger_middle_joint_2',  'robotiq_finger_1_joint_1', 'robotiq_finger_2_joint_1', 'robotiq_finger_middle_joint_3', 'robotiq_finger_1_joint_2',  'robotiq_finger_2_joint_2', 'robotiq_finger_1_joint_3', 'robotiq_finger_2_joint_3'],
             []]

    # Link names for the robots
    links = [['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'camera_link', 'ee_link', 'hand_palm_link', 'hand_link_0_0_link', 'hand_link_12__link', 'hand_link_4_0_link', 'hand_link_8_0_link', 'hand_link_1_0_link', 'hand_link_13__link', 'hand_link_5_0_link', 'hand_link_9_0_link', 'hand_link_2_0_link', 'hand_link_14__link', 'hand_link_6_0_link', 'hand_link_10__link', 'hand_link_3_0_link', 'hand_link_15__link', 'hand_link_7_0_link', 'hand_link_11__link', 'hand_link_3_0_link_tip_link', 'hand_link_15__link_tip_link', 'hand_link_7_0_link_tip_link', 'hand_link_11__link_tip_link', 
              'finger_1_contact_1_link', 'finger_1_contact_2_link', 'finger_1_contact_3_tip_link',
              'finger_2_contact_5_link', 'finger_2_contact_6_link', 'finger_2_contact_7_tip_link',
              'finger_3_contact_9_link', 'finger_3_contact_10_link', 'finger_3_contact_11_tip_link',
              'finger_4_contact_14_link', 'finger_4_contact_15_tip_link'], 
    ['base_link', 'shoulder_link', 'half_arm_1_link', 'half_arm_2_link', 'forearm_link', 'spherical_wrist_1_link', 'spherical_wrist_2_link', 'bracelet_link', 'end_effector_link', 'hand_palm_link', 'hand_link_0_0_link', 'hand_link_12__link', 'hand_link_4_0_link', 'hand_link_8_0_link', 'hand_link_1_0_link', 'hand_link_13__link', 'hand_link_5_0_link', 'hand_link_9_0_link', 'hand_link_2_0_link', 'hand_link_14__link', 'hand_link_6_0_link', 'hand_link_10__link', 'hand_link_3_0_link', 'hand_link_15__link', 'hand_link_7_0_link', 'hand_link_11__link', 'hand_link_3_0_link_tip_link', 'hand_link_15__link_tip_link', 'hand_link_7_0_link_tip_link', 'hand_link_11__link_tip_link', 
     'finger_1_contact_1_link', 'finger_1_contact_2_link', 'finger_1_contact_3_tip_link',
              'finger_2_contact_5_link', 'finger_2_contact_6_link', 'finger_2_contact_7_tip_link',
              'finger_3_contact_9_link', 'finger_3_contact_10_link', 'finger_3_contact_11_tip_link',
              'finger_4_contact_14_link', 'finger_4_contact_15_tip_link'],
    ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'camera_link', 'ee_link'],
    ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'camera_link', 'ee_link']]

    # Fingers tips for the robots
    finger_tips = [["hand_link_8.0_link", "hand_link_0.0_link", "hand_link_4.0_link"],  # ["hand_link_11__link_tip_link", "hand_link_3.0_link_tip_link", "hand_link_7.0_link_tip_link"]
                   ["hand_link_8.0_link", "hand_link_0.0_link", "hand_link_4.0_link"],  # ["hand_link_8.0_link", "hand_link_0.0_link", "hand_link_4.0_link"]
                    ["tool0"],
                    ['tool0']]

    # All joint names
    all_joints = [[], [], [], []]
    all_joints[UR5e] = joints[UR5e] + hand_joints[UR5e]
    all_joints[GEN3] = joints[GEN3] + hand_joints[GEN3]
    all_joints[UR5e_3f] = joints[UR5e_3f] + hand_joints[UR5e_3f]
    all_joints[UR5e_NOGRIP] = joints[UR5e_NOGRIP]



    # ---- Initial pose for the robot ----
    # Initial pose of the robots in quaternions
    ee_init_pose_quat = [[-0.2144, 0.1333, 0.6499, 0.2597, -0.6784, -0.2809, 0.6272],
                         [0.20954, -0.0250, 0.825, -0.6946,  0.2523, -0.6092,  0.2877],
                         [-4.9190e-01,  1.3330e-01,  4.8790e-01,  3.1143e-06, -3.8268e-01,-9.2388e-01,  2.1756e-06],
                         [-4.9190e-01,  1.3330e-01,  4.8790e-01,  3.1143e-06, -3.8268e-01,-9.2388e-01,  2.1756e-06]]
    
    # Obtain Euler angles from the quaternion
    r, p, y = euler_xyz_from_quat(torch.tensor(ee_init_pose_quat)[:, 3:])
    
    euler = torch.cat((r.unsqueeze(-1), p.unsqueeze(-1), y.unsqueeze(-1)), dim=-1).numpy().tolist()
    r = r.numpy().tolist()
    p = p.numpy().tolist()
    y = y.numpy().tolist()

    # Initial pose using Euler angles
    ee_init_pose = torch.cat((torch.tensor(ee_init_pose_quat)[:,:3], torch.tensor(euler)), dim = -1).numpy().tolist()

    # Increments in the original poses for sampling random values on each axis
    ee_pose_incs = [[-0.3,  0.3],
                    [-0.3,  0.3],
                    [-0.3,  0.3],
                    [-0.5,  0.5],
                    [-0.5,  0.5],
                    [-0.5,  0.5]]
    
    # Which robot apply the sampling poses
    apply_range = [False, True, False, False]



    # ---- Target poses ----
    target_pose = [-0.4919, 0.1333, 0.4879, pi, 2*pi, 2.3562]
    target_poses_incs = [[-0.25,  0.25],
                         [-0.25,  0.25],
                         [-0.42,   -0.42],
                         [-2*pi/5*0,  2*pi/5*0],
                         [-2*pi/5*0,  2*pi/5*0],
                         [-pi/2,  pi/2]]
    
    target_poses_incs2 = [[-0.25,  0.25],
                         [-0.25,  0.25],
                         [-0.15,   0.1],
                         [-2*pi/5,  2*pi/5],
                         [-2*pi/5,  2*pi/5],
                         [-2*pi/5,  2*pi/5]]
    # target_poses_incs = [[-0.008,  0.008],
    #                      [-0.008,  0.008],
    #                      [-0.008,  0.008],
    #                      [-0.12,  0.12],
    #                      [-0.12,  0.12],
    #                      [-0.12,  0.12]]
    apply_range_tgt = True

    # Rotations respecto to the end effector robot link frame for object spawning
    # rot_45_z_pos = Rotation.from_rotvec(-pi/4 * np.array([0, 0, 1]))        # Negative 45 degrees rotation in Z axis 
    # rot_225_z_neg = Rotation.from_rotvec(-5*pi/4 * np.array([0, 0, 1]))     # Negative 225 degrees rotation in Z axis 
    # rot_225_z_pos = Rotation.from_rotvec((pi/4 + pi) * np.array([0, 0, 1])) # Positive 225 degrees rotation in Z axis
    # rot_90_x_pos = Rotation.from_rotvec(pi/2 * np.array([1, 0, 0]))         # Positive 90 degrees rotation in X axis

    # Transform to quaternions
    rot_45_z_pos_quat = rot2tensor(rot_45_z_pos).numpy().tolist()
    rot_180_z_pos_quat = rot2tensor(rot_180_z_pos).numpy().tolist()
    # rot_225_z_neg_quat = rot2tensor(rot_225_z_neg).numpy().tolist()
    # rot_225_z_pos_quat = rot2tensor(rot_225_z_pos).numpy().tolist()



    # ---- Reward variables ----
    # reward scales
    rew_scale_dist: float= 1.0
    rew_scale_vel: float= 0.4

    dist_scale = 0.1545
    vel_scale = 1.2104



    # Position threshold for ending the episode
    distance_thres = 0.03 # 0.08 # 0.03
    height_thres = 0.8


    # Bonus for reaching the target
    bonus_tgt_reached = 300
    bonus_lifting = 30
    bonus_close_grip = -2


    # Contacts
    contact_sensors_dict = {}
    contact_matrix = {}



# Function to update the variables in the configuration class
#    using new information in the BimanualDirect class and new number of environments
def update_cfg(cfg, num_envs, device):
    '''
    In:
        - cfg - RLManipulationDirectCfg: configuration class.
        - num_envs - int: number of environments in the simulation.
        - device - str: Cuda or cpu device
    
    Out:
        - cfg - RLManipulationDirectCfg: modified configuration class
    '''

    # cfg.translation_scale = torch.tensor(cfg.translation_scale).to(device)

    cfg.target_pose = torch.tensor(cfg.target_pose).repeat(num_envs, 1).to(device)
    cfg.open = torch.tensor(cfg.open).repeat(num_envs, 1).to(device)
    cfg.close = torch.tensor(cfg.close).repeat(num_envs, 1).to(device)

    cfg.rot_45_z_pos_quat = torch.tensor(cfg.rot_45_z_pos_quat).repeat(num_envs, 1).to(device)
    cfg.rot_180_z_pos_quat = torch.tensor(cfg.rot_180_z_pos_quat).repeat(num_envs, 1).to(device)

    cfg.contact_matrix = cfg.contact_matrix.repeat(num_envs, 1).to(device)

    cfg.moving_joints_gripper = torch.tensor(cfg.moving_joints_gripper).repeat(num_envs, 1).to(device)


    return cfg



# Add the collision sensors to the configuration class according to the number of environments
def update_collisions(cfg, num_envs):




    # Contact between robot 2 finger pads and object
    finger_middle_w_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.robot] + "/robotiq_finger_middle.*",
        update_period=0.001, 
        history_length=1, 
        debug_vis=True,
        filter_prim_paths_expr = [f"/World/envs/env_{i}/Cuboid" for i in range(num_envs)],
    )

    finger_1_w_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.robot] + "/robotiq_finger_1.*",
        update_period=0.001, 
        history_length=1, 
        debug_vis=True,
        filter_prim_paths_expr = [f"/World/envs/env_{i}/Cuboid" for i in range(num_envs)],
    )

    finger_2_w_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.robot] + "/robotiq_finger_2.*",
        update_period=0.001, 
        history_length=1, 
        debug_vis=True,
        filter_prim_paths_expr = [f"/World/envs/env_{i}/Cuboid" for i in range(num_envs)],
    )



    # Dictionary of contact sensors configurations
    cfg.contact_sensors_dict = {"finger_middle_w_object": finger_middle_w_object,
                                "finger_1_w_object": finger_1_w_object,
                                "finger_2_w_object": finger_2_w_object,
                                }
    
    # Updated contact matrix
    cfg.contact_matrix = torch.tensor([2.5 ,2.5,2.5])

    return cfg
