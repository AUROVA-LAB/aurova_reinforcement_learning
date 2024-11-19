from __future__ import annotations

import os
from math import pi
import torch
from collections.abc import Sequence
import numpy as np
from scipy.spatial.transform import Rotation

from omni.isaac.lab_tasks.manager_based.classic.aurova_reinforcement_learning.bimanual_handover.robots_cfg import UR5e_4f_CFG, GEN3_4f_CFG
from .mdp.utils import compute_rewards, save_images_grid

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, combine_frame_transforms
from omni.isaac.lab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat
from omni.isaac.lab.sensors import CameraCfg, Camera, ContactSensorCfg, ContactSensor
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg

'''
                    ############## IMPORTANT #################
   The whole environment is build for two robots: the UR5e and Kinova GEN3-7dof.
   These two variables (cfg.UR5e and cfg.GEN3) serve as an abstraction to treat the robots during the episodes. In fact,
all the methods need an index to differentiate from which robot get the information.
   Also, data storage is performed using lists, not tensors because the joint space of the robots is
different from one another.
'''


@configclass
class BimanualDirectCfg(DirectRLEnvCfg):
    # env
    decimation = 2              # Number of control action updates @ sim dt per policy dt.
    episode_length_s = 1.0      # Length of the episode in seconds
    steps_reset = 40            # Maximum steps in an episode
    angle_scale = pi            # Angle scalation

    num_actions = 7 + 16        # Number of actions per environment (overridden)
    num_observations = 12 + 14  # Number of observations per environment (overridden)

    num_envs = 1                # Number of environments by default (overriden)

    debug_markers = True        # Activate marker visualization
    save_imgs = False           # Activate image saving from cameras
    render_imgs = False         # Activate image rendering
    render_steps = 6            # Render images every certain amount of steps

    velocity_limit = 10         # Velocity limit for robots' end effector

    # simulation
    sim: SimulationCfg = SimulationCfg(dt = 1/120, render_interval = decimation)
    # SimulationCfg: configuration for simulation physics 
    #    dt: time step of the simulation (seconds)
    #    render_interval: number of physics steps per rendering steps

    UR5e = 0
    GEN3 = 1

    keys = ['UR5e', 'GEN3']     # Keys for the robots in simulation
    ee_link = ['tool0',         # Names of the end effector
               'tool_frame']

    # robots
    robot_cfg_1: Articulation = UR5e_4f_CFG.replace(prim_path="/World/envs/env_.*/" + keys[UR5e])
    robot_cfg_2: Articulation = GEN3_4f_CFG.replace(prim_path="/World/envs/env_.*/" + keys[GEN3])
    
    # Robot joint names
    joints = [['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
              ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']]
    
    # Hand joint names
    hand_joints = [['joint_' + str(i) + '_0' for i in range(0,16)] for i in range(2)]

    links = [['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'camera_link', 'ee_link', 'hand_palm_link', 'hand_link_0_0_link', 'hand_link_12__link', 'hand_link_4_0_link', 'hand_link_8_0_link', 'hand_link_1_0_link', 'hand_link_13__link', 'hand_link_5_0_link', 'hand_link_9_0_link', 'hand_link_2_0_link', 'hand_link_14__link', 'hand_link_6_0_link', 'hand_link_10__link', 'hand_link_3_0_link', 'hand_link_15__link', 'hand_link_7_0_link', 'hand_link_11__link', 'hand_link_3_0_link_tip_link', 'hand_link_15__link_tip_link', 'hand_link_7_0_link_tip_link', 'hand_link_11__link_tip_link'], 
    ['base_link', 'shoulder_link', 'half_arm_1_link', 'half_arm_2_link', 'forearm_link', 'spherical_wrist_1_link', 'spherical_wrist_2_link', 'bracelet_link', 'end_effector_link', 'hand_palm_link', 'hand_link_0_0_link', 'hand_link_12__link', 'hand_link_4_0_link', 'hand_link_8_0_link', 'hand_link_1_0_link', 'hand_link_13__link', 'hand_link_5_0_link', 'hand_link_9_0_link', 'hand_link_2_0_link', 'hand_link_14__link', 'hand_link_6_0_link', 'hand_link_10__link', 'hand_link_3_0_link', 'hand_link_15__link', 'hand_link_7_0_link', 'hand_link_11__link', 'hand_link_3_0_link_tip_link', 'hand_link_15__link_tip_link', 'hand_link_7_0_link_tip_link', 'hand_link_11__link_tip_link']]

    # All agent joint names
    all_joints = [[], []]
    all_joints[UR5e] = joints[UR5e] + hand_joints[UR5e]
    all_joints[GEN3] = joints[GEN3] + hand_joints[GEN3]
    

    # contact sensors
    # Contact between robot 1 hand and object
    object_w_hands: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Cuboid", 
        update_period=0.1, 
        history_length=2, 
        debug_vis=True,
        filter_prim_paths_expr =[]  # Bad declared on purpose, corrected later on
    )
    # ContactSensorCfg: Configuration for the contact sensor.
    #    update_period: Update period of the sensor buffers (in seconds).
    #    history_length: Number of past frames to store in the sensor buffers.
    #    debug_vis: Whether to visualize the sensor.
    #    filter_prim_paths_expr: The list of primitive paths (or expressions) to filter contacts with.
    #        It is declared as a list because it does not work using "regex" expressions

    # Dictionary of contact sensors configurations --> Updated later
    contact_sensors_dict = {"object_w_hands": object_w_hands}


    # camera
    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 1.0e6)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-1.58,  -0.11711,  0.28), rot=(-0.5, -0.5, -0.5, -0.5), convention="ros"),
    )
    # CameraCfg: Configuration for a camera sensor.
    #    update_period: Update period of the sensor buffers (in seconds).
    #    width: Width of the image in pixels.
    #    height: Height of the image in pixels.
    #    data_types: List of sensor names/types to enable for the camera.
    #    spawn: Spawn configuration for the asset.
    #    offset: The offset pose of the sensor's frame from the sensor's parent frame.


    # object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cuboid",

        spawn=sim_utils.CuboidCfg(
            size=(0.035, 0.035, 0.35),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01655),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled = True),
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


    # markers
    marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "cmd_ur5e": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
                visible = debug_markers
            ),
            "ur5e_ee_pose": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
                visible = debug_markers
            ),
            "gen3_ee_pose": sim_utils.UsdFileCfg(
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
    

    # Traslation respect to the end effector robot link frame for object spawning
    obj_pos_trans = torch.tensor([0.0, -0.0335*2, 0.115])
    '''
    X: Positivo (diagonal hacia arriba)
    Y: Negativo (diagonal hacia abajo)
    Z: En el eje longitudinal
    '''
    
    # Rotations respecto to the end effector robot link frame for object spawning
    rot_45_z_neg = Rotation.from_rotvec(-pi/4 * np.array([0, 0, 1]))        # Negative 45 degrees rotation in Z axis 
    rot_90_x_pos = Rotation.from_rotvec(pi/2 * np.array([1, 0, 0]))         # Positive 90 degrees rotation in X axis

    # Aggregate rotations as quaternions
    rot_quat = torch.tensor((rot_45_z_neg*rot_90_x_pos).as_quat())

    # In SCIPY, the real value (w) of a quaternion is at [-1] position, 
    #    but for IsaacLab it needs to be in [0] position 
    obj_quat_trans = torch.zeros((4))
    obj_quat_trans[0], obj_quat_trans[1:] = rot_quat[-1].clone(), rot_quat[:3].clone()
    

    # Initial pose of the robots in quaternions
    ee_init_pose_quat = torch.tensor([[-0.5144, 0.1333, 0.6499, 0.2597, -0.6784, -0.2809, 0.6272], 
                                      [0.2954, -0.0250, 0.5662, -0.2845, -0.6176, -0.2554, -0.6873]])
    
    # Obtain Euler angles from the quaternion
    r, p, y = euler_xyz_from_quat(ee_init_pose_quat[:, 3:])
    euler = torch.cat((r.unsqueeze(-1), p.unsqueeze(-1), y.unsqueeze(-1)), dim=-1)

    # Initial pose using Euler angles
    ee_init_pose = torch.cat((ee_init_pose_quat[:,:3], euler), dim = -1)

    # Increments in the original poses for sampling random values on each axis
    ee_pose_incs = torch.tensor([[-0.2,  0.2],
                                 [-0.2,  0.2],
                                 [-0.2,  0.2],
                                 [-0.8,  0.8],
                                 [-0.8,  0.8],
                                 [-0.8,  0.8]])
    
    # Translation respect to the object link frame for object grasping point observation
    grasp_obs_obj_pos_trans = torch.tensor([0.0, 0.0, 0.1])
    grasp_obs_obj_quat_trans = torch.tensor([1.0, 0.0, 0.0, 0.0])

    # reward scales
    rew_position_tracking: float = -0.2
    rew_position_tracking_fine_grained : float= 0.1
    rew_orientation_tracking: float = -0.1
    rew_dual_quaternion_error: float= 0.0
    rew_action_rate: float= -0.0001
    rew_joint_vel: float= -0.0001


# Function to update the variables in the configuration class
#    using new information in the BimanualDirect class
def update_cfg(cfg, num_envs, device):
    '''
    In:
        - cfg - BimanualDirectCfg: configuration class.
        - num_envs - int: number of environments in the simulation.
        - device - str: Cuda or cpu device
    
    Out:
        - cfg - BimanualDirectCfg: modified configuration class
    '''
    cfg.obj_pos_trans = cfg.obj_pos_trans.repeat(num_envs, 1).to(device)
    cfg.obj_quat_trans = cfg.obj_quat_trans.repeat(num_envs, 1).to(device)

    cfg.grasp_obs_obj_pos_trans = cfg.grasp_obs_obj_pos_trans.repeat(num_envs, 1).to(device)
    cfg.grasp_obs_obj_quat_trans = cfg.grasp_obs_obj_quat_trans.repeat(num_envs, 1).to(device)

    return cfg


def update_collisions(cfg, num_envs):

    # Contact between robot 1 hand and object
    robot1_w_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.UR5e] + "/.*_link",
        update_period=0.1, 
        history_length=2, 
        debug_vis=True,
        filter_prim_paths_expr = [f"/World/envs/env_{i}/Cuboid" for i in range(num_envs)],
    )

    # Contact between robot 2 hand and object
    robot2_w_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.GEN3] + "/hand_.*",
        update_period=0.1, 
        history_length=2, 
        debug_vis=True,
        filter_prim_paths_expr = [f"/World/envs/env_{i}/Cuboid" for i in range(num_envs)],
    )


    # Contact between robot 2 hand and object
    robot1_w_robot2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.GEN3] + "/.*_link",
        update_period=0.1, 
        history_length=2, 
        debug_vis=True,
        filter_prim_paths_expr = [f"/World/envs/env_{i}/{cfg.keys[cfg.UR5e]}/{joint}" for i in range(cfg.num_envs) for joint in cfg.links[cfg.UR5e]],
    )
    print([f"/World/envs/env_{i}/{cfg.keys[cfg.UR5e]}/{joint}" for i in range(cfg.num_envs) for joint in cfg.links[cfg.UR5e]])

    # print([f"/World/envs/env_{i}/{cfg.keys[cfg.UR5e]}/{joint}" for i in range(cfg.num_envs) for robot in cfg.links for joint in robot])

    cfg.contact_sensors_dict = {"robot1_w_object": robot1_w_object,
                                "robot2_w_object": robot2_w_object,
                                "robot1_w_robot2": robot1_w_robot2} 
    

    return cfg