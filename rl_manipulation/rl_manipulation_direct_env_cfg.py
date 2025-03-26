from __future__ import annotations

import os
from math import pi
import torch
from collections.abc import Sequence
import numpy as np
from scipy.spatial.transform import Rotation

from omni.isaac.lab_tasks.manager_based.classic.aurova_reinforcement_learning.rl_manipulation.robots_cfg import UR5e_4f_CFG, UR5e_3f_CFG, GEN3_4f_CFG
from .mdp.utils import compute_rewards, save_images_grid

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import euler_xyz_from_quat
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
rot_45_z_neg = Rotation.from_rotvec(-pi/4 * np.array([0, 0, 1]))        # Negative 45 degrees rotation in Z axis 
rot_305_z_neg = Rotation.from_rotvec(-5*pi/4 * np.array([0, 0, 1]))     # Negative 135 degrees rotation in Z axis 
rot_45_z_pos = Rotation.from_rotvec((pi/4 + pi) * np.array([0, 0, 1]))
rot_90_x_pos = Rotation.from_rotvec(pi/2 * np.array([1, 0, 0]))         # Positive 90 degrees rotation in X axis


@configclass
class EventCfg:
  robot_physics_material = EventTermCfg(
      func=mdp.randomize_rigid_body_material,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg("UR5e_3f", body_names=".*"),
          "static_friction_range": (0.7, 1.3),
          "dynamic_friction_range": (1.0, 1.0),
          "restitution_range": (1.0, 1.0),
          "num_buckets": 250,
      },
  )
  robot_joint_stiffness_and_damping = EventTermCfg(
      func=mdp.randomize_actuator_gains,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg("UR5e_3f", joint_names=".*"),
          "stiffness_distribution_params": (0.75, 1.5),
          "damping_distribution_params": (0.3, 3.0),
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

  body_mass = EventTermCfg(
      func = mdp.randomize_rigid_body_mass,
      mode = "reset",
      params = {
          "asset_cfg": SceneEntityCfg("object"),
          "mass_distribution_params": (0.5, 1.5),
          "operation": "scale",
          "distribution": "uniform",
          "recompute_inertia": False,
      }
  )

# Configuration class for the environment
@configclass
class RLManipulationDirectCfg(DirectRLEnvCfg):

    # events: EventCfg = EventCfg()
    
    # ---- Env variables ----
    decimation = 3              # Number of control action updates @ sim dt per policy dt.
    episode_length_s = 3.0      # Length of the episode in seconds
    max_steps = 275             # Maximum steps in an episode
    angle_scale = 5*pi/180.0    # Action angle scalation
    translation_scale = torch.tensor([0.02, 0.02, 0.02]) # Action translation scalation
    hand_joint_scale = 0.075    # Hand joint scalation

    # Variables to distinguish the phases
    APPROACH = 0
    MANIPULATION = 1

    phase = MANIPULATION       # Phase of the task (0: approach, 1: manipulation)
    option = 0                 # Option for the NN (0: everything, 1: pre-trained MLP, 2: pre-trained MLP with GNN)

    path_to_pretrained = "2024-12-11_11-04-13/model_53248000_steps" # Path to the pre-trained approaching model

    num_actions = 6 + phase * 1          # Number of actions per environment (overridden)
    num_observations = 7 + 7 + phase * (1)     # Number of observations per environment (overridden)
    euler_flag = True                     # Wether to use Euler angles or quaternions for the actions

    num_envs = 1                # Number of environments by default (overriden)

    debug_markers = True        # Activate marker visualization
    save_imgs = False           # Activate image saving from cameras
    render_imgs = True          # Activate image rendering
    render_steps = 6            # Render images every certain amount of steps

    velocity_limit = 10         # Velocity limit for robots' end effector

    seq_len = 3                 # Length of the sequence

    channels = 3                # Number of channels
    height = 480
    width = 640

    UR5e = 0
    GEN3 = 1
    UR5e_3f = 2
    
    robot = UR5e_3f

    keys = ['UR5e', 'GEN3', 'UR5e_3f']     # Keys for the robots in simulation
    ee_link = ['tool0',         # Names for the end effector of each robot
               'tool_frame',
               'tool0']



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
    

    # Object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",

        spawn=sim_utils.CylinderCfg(
            # size=(0.035, 0.035, 0.45),
            radius = 0.05,
            height = 0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.00025),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled = True,
                                                            contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos = [-1, -0.11711,  0.05], rot=[0.4849,  0.4849,  0.5146,  0.5146]),
    )# 0.7071,  0.7071, -0.0012, -0.0012
    # RigidObjectCfg: Configuration parameters for a rigid object.
    #    spawn: Spawn configuration for the asset. --> Deciding which object type it is spawned
    #       CylinderCfg: Configuration parameters for a Cylinder prim.
    #          size: Size of the Cylinder.
    #          rigid_props / mass_props / collision_props / visual_material: properties of the prim declaration
    #    init_state: Initial state of the rigid object. --> Initial pose

    # Markers
    marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "ur5e_ee_pose": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
                visible = debug_markers
            ),
            "grasp_point_obj": sim_utils.UsdFileCfg(
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
    
    # camera
    camera_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/" + keys[robot] + "/camera_depth_frame/Camera",
        update_period=0.03,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.86, focus_distance=400.0, horizontal_aperture=8.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )


    # ---- Joint information ----
    # Robot joint names
    joints = [['arm_shoulder_pan_joint', 'arm_shoulder_lift_joint', 'arm_elbow_joint', 'arm_wrist_1_joint', 'arm_wrist_2_joint', 'arm_wrist_3_joint'],
              ['arm_joint_1', 'arm_joint_2', 'arm_joint_3', 'arm_joint_4', 'arm_joint_5', 'arm_joint_6', 'arm_joint_7'],
              ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']]
    
    # Hand joint names
    hand_joints = [['joint_' + str(i) + '_0' for i in range(0,16)] for i in range(2)] + \
            [["robotiq_finger_1_joint_1", "robotiq_finger_1_joint_2", "robotiq_finger_1_joint_3",
             "robotiq_finger_2_joint_1", "robotiq_finger_2_joint_2", "robotiq_finger_2_joint_3",
             "robotiq_finger_middle_joint_1", "robotiq_finger_middle_joint_2", "robotiq_finger_middle_joint_3"]]

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
    ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'camera_link', 'ee_link']]

    # Fingers tips for the robots
    finger_tips = [["hand_link_8.0_link", "hand_link_0.0_link", "hand_link_4.0_link"],  # ["hand_link_11__link_tip_link", "hand_link_3.0_link_tip_link", "hand_link_7.0_link_tip_link"]
                   ["hand_link_8.0_link", "hand_link_0.0_link", "hand_link_4.0_link"],  # ["hand_link_8.0_link", "hand_link_0.0_link", "hand_link_4.0_link"]
                    ["tool0"]]
    
    # Displacement from the tips
    tips_displacement = torch.tensor([0.03, -0.03, 0.0])

    # All joint names
    all_joints = [[], [], []]
    all_joints[UR5e] = joints[UR5e] + hand_joints[UR5e]
    all_joints[GEN3] = joints[GEN3] + hand_joints[GEN3]
    all_joints[UR5e_3f] = joints[UR5e_3f] + hand_joints[UR5e_3f]

    m1 = 1.2218 / 140



    # ---- Collision information ----
    # Dictionary of contact sensors configurations --> Updated later
    contact_sensors_dict = {}

    # Contact matrix for weight the contacts
    contact_matrix = torch.tensor([[0.0]])



    # ---- Initial pose for the robot ----
    # Initial pose of the robots in quaternions
    ee_init_pose_quat = torch.tensor([[-0.2144, 0.1333, 0.6499, 0.2597, -0.6784, -0.2809, 0.6272],  #   0.63,0.28,-0.68,-0.26
                                      [0.20954, -0.0250, 0.825, -0.6946,  0.2523, -0.6092,  0.2877],
                                      [-4.9190e-01,  1.3330e-01,  4.8790e-01,  3.1143e-06, -3.8268e-01,-9.2388e-01,  2.1756e-06]])
    
    # Obtain Euler angles from the quaternion
    r, p, y = euler_xyz_from_quat(ee_init_pose_quat[:, 3:])
    euler = torch.cat((r.unsqueeze(-1), p.unsqueeze(-1), y.unsqueeze(-1)), dim=-1)

    # Initial pose using Euler angles
    ee_init_pose = torch.cat((ee_init_pose_quat[:,:3], euler), dim = -1)

    # Increments in the original poses for sampling random values on each axis
    ee_pose_incs = torch.tensor([[-0.15,  0.15],
                                 [-0.15,  0.15],
                                 [-0.15,  0.15],
                                 [-0.3,  0.3],
                                 [-0.3,  0.3],
                                 [-0.3,  0.3]])
    
    # Which robot apply the sampling poses
    apply_range = [False, True, False]



    # ---- Object poses ----
    # Traslation respect to the end effector robot link frame for object spawning
    obj_pos_trans = torch.tensor([0.0 - 0.075, -0.0335*2 - 0.075, 0.115])

    # Transform to quaternions
    rot_45_z_neg_quat = rot2tensor(rot_45_z_neg)
    rot_305_z_neg_quat = rot2tensor(rot_305_z_neg)
    rot_45_z_pos_quat = rot2tensor(rot_45_z_pos)

    # Aggregate rotations as quaternions
    rot_quat = torch.tensor((rot_45_z_neg*rot_90_x_pos).as_quat())

    # In SCIPY, the real value (w) of a quaternion is at [-1] position, 
    #    but for IsaacLab it needs to be in [0] position 
    obj_quat_trans = torch.zeros((4))
    obj_quat_trans[0], obj_quat_trans[1:] = rot_quat[-1].clone(), rot_quat[:3].clone()
    
    # Height limits for the object and the GEN3 robot
    object_height_limit = ee_init_pose_quat[0, 2] + ee_pose_incs[0, 0] - 0.15 # = 0.45
    gen3_height_limit = 0.1
    
    # Translation respect to the object link frame for object grasping point observation
    grasp_obs_obj_pos_trans = torch.tensor([0.0, 0.0, 0.0])
    grasp_obs_obj_quat_trans = rot2tensor(rot_90_x_pos)

    # Target position for the object -> origin GEN3 position with offset in X axis
    target_pose = torch.tensor([0.1054, -0.0250, 0.5662, -0.2845, -0.6176, -0.2554, -0.6873])
    

    obj_pose = torch.tensor([-0.5, 0, 0.15, 1,0,0,0])
    # obj_pose = torch.cat((obj_pose, rot2tensor(a)))

    obj_poses_incs = torch.tensor([[-0.15,  0.15],
                                 [-0.15,  0.15],
                                 [-0.0,  0.0],
                                 [0.0,  0.0],
                                 [0.0,  0.0],
                                 [-pi,  pi]])



    # ---- Reward variables ----
    # reward scales
    rew_scale_hand_obj: float= 1.0
    rew_scale_obj_target: float= 12.0

    # Position threshold for changing reach reward
    rew_change_thres = 0.0235 # 0.018
    obj_reach_target_thres = 0.01

    # Bonus for reaching the target
    bonus_obj_reach = 300





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

    cfg.translation_scale = cfg.translation_scale.to(device)

    cfg.obj_pos_trans = cfg.obj_pos_trans.repeat(num_envs, 1).to(device)
    cfg.obj_quat_trans = cfg.obj_quat_trans.repeat(num_envs, 1).to(device)

    cfg.grasp_obs_obj_pos_trans = cfg.grasp_obs_obj_pos_trans.repeat(num_envs, 1).to(device)
    cfg.grasp_obs_obj_quat_trans = cfg.grasp_obs_obj_quat_trans.repeat(num_envs, 1).to(device)

    cfg.obj_pose = cfg.obj_pose.repeat(num_envs, 1).to(device)
    cfg.target_pose = cfg.target_pose.repeat(num_envs, 1).to(device)

    cfg.rot_45_z_neg_quat = cfg.rot_45_z_neg_quat.repeat(num_envs, 1).to(device)
    cfg.rot_305_z_neg_quat = cfg.rot_305_z_neg_quat.repeat(num_envs, 1).to(device)
    cfg.rot_45_z_pos_quat = cfg.rot_45_z_pos_quat.repeat(num_envs, 1).to(device)

    cfg.tips_displacement = cfg.tips_displacement.repeat(num_envs, 1).to(device)

    cfg.contact_matrix = cfg.contact_matrix.to(device)
    
    return cfg




# Add the collision sensors to the configuration class according to the number of environments
def update_collisions(cfg, num_envs):

    # Contact between robot tips and object
    tip_1_w_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.robot] + "/robotiq_finger_1_link_3",
        update_period=0.001, 
        history_length=1, 
        debug_vis=True,
        filter_prim_paths_expr = [f"/World/envs/env_{i}/Object" for i in range(num_envs)],
    )

    tip_2_w_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.robot] + "/robotiq_finger_2_link_3",
        update_period=0.001, 
        history_length=1, 
        debug_vis=True,
        filter_prim_paths_expr = [f"/World/envs/env_{i}/Object" for i in range(num_envs)],
    )

    tip_middle_w_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.robot] + "/robotiq_finger_middle_link_3",
        update_period=0.001, 
        history_length=1, 
        debug_vis=True,
        filter_prim_paths_expr = [f"/World/envs/env_{i}/Object" for i in range(num_envs)],
    )

    # Contact between robot 2 and the ground
    robot_w_ground: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.robot] + "/.*_link",
        update_period=0.001, 
        history_length=1, 
        debug_vis=True,
        filter_prim_paths_expr = ["/World/ground/GroundPlane/CollisionPlane"],
    )


    # Dictionary of contact sensors configurations
    cfg.contact_sensors_dict = {
                                "tip_1_w_object": tip_1_w_object,
                                "tip_2_w_object": tip_2_w_object,
                                "tip_middle_w_object": tip_middle_w_object,

                                "robot_w_ground": robot_w_ground
                                }
    
    # Updated contact matrix
    cfg.contact_matrix = torch.tensor([0.65, 0.65, 0.65,
                                        -2.0])


    return cfg
