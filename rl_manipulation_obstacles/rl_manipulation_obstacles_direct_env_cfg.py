from __future__ import annotations

from math import pi
import torch
import copy
import numpy as np
from scipy.spatial.transform import Rotation

from isaaclab_tasks.manager_based.aurova_reinforcement_learning.rl_manipulation_obstacles.robots_cfg import *


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, subtract_frame_transforms, combine_frame_transforms
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg



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


rot_180_z_pos = Rotation.from_rotvec(pi/2 * np.array([0, 1, 0]))        # Positive 180 degrees rotation in Z axis 
rot_45_z_pos = Rotation.from_rotvec((pi/4) * np.array([0.05, 0.05, 1.05]))     # Positive 45 degrees rotation in Z axis 



# Configuration class for the environment
@configclass
class RLManipulationObstaclesDirectCfg(DirectRLEnvCfg):
    
    # ---- Env variables ----
    decimation = 1              # Number of control action updates @ sim dt per policy dt.
    episode_length_s = 2.0      # Length of the episode in seconds
    max_steps = 400             # Maximum steps in an episode
   
    # --- Mapping configuration ---
    DQ = 0
    EULER = 1
    QUAT = 2
    MAT = 3

    # Size of the Lie algebra
    sizes = [[8, 6, 7, 16], [6]*4]
    
    representation = DQ
    mapping = 1
    size = sizes[int(mapping != 0)][representation]
    size_group = sizes[0][representation]
    distance = 1

    # Scalings for each action
    scalings = [[[0.01, 0.001], [0.03,  0.006], [0.01, 0.007]],
                [[0.007, 0.02]],
                [[0.006, 0.025], [0.006, 0.03], [0.007, 0.015], [0.007, 0.015]],
                [[0.02,  0.004], [0.03,  0.006]]]

    action_scaling = scalings[representation][mapping]
    grip_scaling = 5*2

    img_width, img_height = 128, 128# 640, 480


    # --- Action / observation space ---
    action_space = 7             # Number of actions per environment (overridden)
    observation_space = 6 + 1 + 3#  + img_height*img_width*3       # Number of observations per environment (overridden)
    state_space = observation_space

    num_envs = 1                # Number of environments by default (overriden)

    debug_markers = False       # Activate marker visualization
    save_imgs = False           # Activate image saving from cameras
    render_steps = 6            # Render images every certain amount of steps

    velocity_limit = 10         # Velocity limit for robots' end effector


    # Robot options
    UR5e = 0                    
    GEN3 = 1
    UR5e_3f = 2
    UR5e_NOGRIP = 3
    
    robot = UR5e_3f

    keys = ['UR5e', 'GEN3', 'UR5e_3f', 'UR5e_NOGRIP']     # Keys for the robots in simulation
    ee_link = ['tool0',         # Names for the end effector of each robot
               'tool_frame',
               'tool0',
               'wrist_3_link']
    
     # Robotiq 3f control
    grip_theta_max = [1.2218, 1.5708]
    m = [grip_theta_max[0]/140, grip_theta_max[1]/100]

    def_pos = [0.0, 0.05, 0.0, -0.053]

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

    moving_joints_gripper = [m[0], 
                             m[0],
                             m[0], 0*m[0],
                             
                             -m[0]*0,
                             0.0, 0.0,
                             -m[0]*0, -m[0]*0]



    # ---- Configurations ----
    # Simulation
    sim: SimulationCfg = SimulationCfg(dt = 1/100, render_interval = decimation)
    # SimulationCfg: configuration for simulation physics 
    #    dt: time step of the simulation (seconds)
    #    render_interval: number of physics steps per rendering steps

    # Robots
    robot_cfg_1: Articulation = UR5e_4f_CFG.replace(prim_path="/World/envs/env_.*/" + keys[UR5e])
    robot_cfg_2: Articulation = GEN3_4f_CFG.replace(prim_path="/World/envs/env_.*/" + keys[GEN3])
    robot_cfg_3: Articulation = UR5e_3f_CFG.replace(prim_path="/World/envs/env_.*/" + keys[UR5e_3f])
    robot_cfg_4: Articulation = UR5e_NOGRIP_CFG.replace(prim_path="/World/envs/env_.*/" + keys[UR5e_NOGRIP])

    shelf_cfg: RigidObject = SHELF.replace(prim_path="/World/envs/env_.*/shelf")
    # object_cfg: RigidObject = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/object",

    #     spawn=sim_utils.CylinderCfg(
    #         radius = 0.05,
    #         height = 0.31,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity = False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.000025),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled = True,
    #                                                         contact_offset=0.001),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos = [-1, -0.11711,  0.05]),
    # )
    object_cfg: RigidObject = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",

        spawn=sim_utils.CuboidCfg(
            size = [0.075, 0.25, 0.075],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity = False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.00025),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled = True,
                                                            contact_offset=0.015),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos = [-1, -0.11711,  0.05]),
    )
    # object_cfg: RigidObject = MASTER_CHEF_CAN.replace(prim_path="/World/envs/env_.*/object")

    
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
            "interm_point": sim_utils.UsdFileCfg(
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

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/camera",

        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 5.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),

        data_types=["rgb", "depth", "instance_id_segmentation_fast"],

        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.1,              # ← computed
            horizontal_aperture=20.955,     # ← assumed
            clipping_range=(0.1, 1.5),
        ),

        width=img_width,
        height=img_height,

        depth_clipping_behavior="max",
        colorize_instance_id_segmentation=True,
    )

    tiled_camera_ext: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/camera_ext",

        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 5.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),

        data_types=["rgb", "depth", "instance_id_segmentation_fast"],

        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.1,              # ← computed
            horizontal_aperture=20.955,     # ← assumed
            clipping_range=(0.1, 1.5),
        ),

        width=img_width,
        height=img_height,

        depth_clipping_behavior="max",
        colorize_instance_id_segmentation=True,
    )

    tiled_camera_front: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/camera_front",

        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 5.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),

        data_types=["rgb", "depth", "instance_id_segmentation_fast"],

        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.1,              # ← computed
            horizontal_aperture=20.955,     # ← assumed
            clipping_range=(0.1, 1.5),
        ),

        width=img_width,
        height=img_height,

        depth_clipping_behavior="max",
        colorize_instance_id_segmentation=True,
    )



    camera_trans = [[0.1231539748184402, 0.09738024537036244, 0.015012247696052522]]
    camera_rot = [[-0.3825884841399441, -0.00019676447364075367, -0.00034948181171445825, -0.9239187685882916]]

    # D_origin = 1.28 + 0.1
    camera_ext_trans = [[-0.250333604818459999, 0.70019047623759998, 0.6752149041133910001]] # [[-0.5333604818459999, 1.0019047623759998, 0.7149041133910001]]
    camera_ext_rot = [[0.016458520000000004, 0.02048439000000002, 0.89798281, -0.43919778]]

    # camera_ext_trans = [[0.2778,  1.1144,  1.2721]]
    # camera_ext_rot = [[1.0, 0.0, 0.0, 0.0]]

    rot_neg90_xy = torch.tensor([(Rotation.from_rotvec(-pi/2 * np.array([-1, 1, 0]))).as_quat()])               # Negative 90 degrees rotation in Y axis 
    rot_neg90_xy[:, 0], rot_neg90_xy[:, 1:] = rot_neg90_xy.clone()[:, 3], rot_neg90_xy.clone()[:, :3]
    rot_neg90_xy = rot_neg90_xy.numpy().tolist()



    camera_ext_trans_front = [[-0.85,  0.13,  0.6]]
    camera_ext_rot_front = [[1.0, 0.0, 0.0, 0.0]]

    rot_neg90_xy_2 = torch.tensor([(Rotation.from_rotvec(1.4*pi/2 * np.array([0, 1, 0]))).as_quat()])               # Negative 90 degrees rotation in Y axis 
    rot_neg90_xy_2[:, 0], rot_neg90_xy_2[:, 1:] = rot_neg90_xy_2.clone()[:, 3], rot_neg90_xy_2.clone()[:, :3]
    rot_neg90_xy_2 = rot_neg90_xy_2.numpy().tolist()

    rot_neg90_xy_3 = torch.tensor([(Rotation.from_rotvec(-pi/2 * np.array([0, 0, 1]))).as_quat()])               # Negative 90 degrees rotation in Y axis 
    rot_neg90_xy_3[:, 0], rot_neg90_xy_3[:, 1:] = rot_neg90_xy_3.clone()[:, 3], rot_neg90_xy_3.clone()[:, :3]
    rot_neg90_xy_3 = rot_neg90_xy_3.numpy().tolist()



    # ---- Joint information ----
    # Robot joint names
    joints = [['arm_shoulder_pan_joint', 'arm_shoulder_lift_joint', 'arm_elbow_joint', 'arm_wrist_1_joint', 'arm_wrist_2_joint', 'arm_wrist_3_joint'],
              ['arm_joint_1', 'arm_joint_2', 'arm_joint_3', 'arm_joint_4', 'arm_joint_5', 'arm_joint_6', 'arm_joint_7'],
              ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
              ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']]
    
    # Hand joint names
    hand_joints = [['joint_' + str(i) + '_0' for i in range(0,16)] for i in range(2)] + \
            [["robotiq_finger_1_joint_1", "robotiq_finger_1_joint_2", "robotiq_finger_1_joint_3",
             "robotiq_finger_2_joint_1", "robotiq_finger_2_joint_2", "robotiq_finger_2_joint_3",
             "robotiq_finger_middle_joint_1", "robotiq_finger_middle_joint_2", "robotiq_finger_middle_joint_3",],
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



    # ---- Initial pose for the robot OBSTACLES----
    # Initial pose of the robots in quaternions
    # ee_init_pose_quat = [[-0.2144, 0.1333, 0.6499, 0.2597, -0.6784, -0.2809, 0.6272],
    #                      [0.20954, -0.0250, 0.825, -0.6946,  0.2523, -0.6092,  0.2877],
    #                      [-0.1030,  0.1225,  0.7802,  -0.2031, 0.6846, 0.1954,  -0.6722],
    #                      [-0.2019,  0.1292,  0.6284, -0.4223, -0.2331, -0.8642,  0.1430]]
    
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
                    [-0.7 / 3 ,  0.7 / 3 ],
                    [-0.7 / 3 ,  0.7 / 3 ],
                    [-0.7 / 3 ,  0.7 / 3 ]]
    
    # Which robot apply the sampling poses
    apply_range = [False, False, True, False]

    ee_translation = [0.0, 0.0, 0.25]
    ee_rotation = [1.0, 0.0, 0.0, 0.0]



    # ---- Target poses for obstacles ----
    target_pose = [-0.4919, 0.1333, 0.4879, pi, 2*pi, 2.3562]
    target_poses_incs = [[-0.2,  0.1],
                         [-0.2,   0.2],
                         [-0.225,   0.225],
                         [-pi/4,  pi/4],
                         [-pi/4,  pi/4],
                         [-pi/4,  pi/4]]

    apply_range_tgt = True

    box_range_x = [0,0]
    box_range_y = [1,1]
    object_base_pose = [-0.6800, -0.3700,  0.1400,  1.0000,  0.0000,  0.0000,  0.0000]
    object_increments = [0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0]


    # Shelf poses
    p1 = [-0.75, -0.6, 0.25, 1,0,0,0]
    p2 = [-0.75, -0.35, 0.0, 1,0,0,0]

    obst_list = []
    ellipsoid_r = []
    sel = p1

    # for i in range(2):
    #     if i == 0:
    #         obst_list.append(p1)
    #         p_ = copy.deepcopy(p1)
    #     else:
    #         obst_list.append(p2)
    #         p_ = copy.deepcopy(p2)

    #     for j in range(4):

    #         if j != 0:
    #             p_[1] += 0.5
    #             obst_list.append(copy.deepcopy(p_))
            
    #         p__ = copy.deepcopy(p_)

    #         for k in range(3):
    #             p__[2] += 0.5
    #             obst_list.append(copy.deepcopy(p__))


    # ellipsoid_r = [0.43, 
    #                0.2, 
    #                0.435 ]
    
    # ---- MPC Configuration -----
    n_steps_mpc = 200
    path_traj_mpc = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/aurova_reinforcement_learning/rl_manipulation_obstacles/trajectories"

    lie_mpc = True
    dt = 0.1

    get_img_mpc = False
    get_rot = True

    plan_chg_thres = 0.01

    test = False
    model_path = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/aurova_reinforcement_learning/rl_manipulation_obstacles/train/sam2/best_model.pth"
    
    save_interval = 3

    mode = "pcd"



    # ---- Target poses for Pick-and-Place ----
    # ---- Target poses ----
    target_pose = [-0.4919, 0.1333, 0.04, 0, 3, 0]
    target_pose_2 = [-0.4308,  0.1459,  0.4802-0.25,  0, 3, 0]
    # target_pose_2 = [-0.4308,  0.1459,  0.4802-0.25,  3.1350, -0.1133, 2.2588]

    target_poses_incs = [[-0.2,  0.2],
                         [-0.2,   0.2],
                         [-0.1*0,   0.1*0],
                         [-2*pi/5*0,  2*pi/5*0],
                         [-2*pi/5*0,  2*pi/5*0],
                         [-pi/2,  pi/2]]
    
    target_poses_incs2 = [[-0.1*0,  0.1*0],
                          [-0.1*0,  0.1*0],
                          [-0.1*0,  0.1*0],
                          [-1*pi/5*0,  1*pi/5*0],
                          [-1*pi/5*0,  1*pi/5*0],
                          [-1*pi/5*0,  1*pi/5*0]]

    apply_range_tgt = True
                

    rot_45_z_pos_quat = rot2tensor(rot_45_z_pos).numpy().tolist()
    rot_180_z_pos_quat = rot2tensor(rot_180_z_pos).numpy().tolist()

    # Object pose adjustments
    object_translation = torch.tensor([np.array([0.0, 0.0, 0.1])])
    rot_neg90_y = torch.tensor([(Rotation.from_rotvec(-pi/2 * np.array([0, 1, 0]))).as_quat()])               # Negative 90 degrees rotation in Y axis 
    rot_pos135_z = torch.tensor([(Rotation.from_rotvec((pi/2+pi/4) * np.array([0, 0, 1]))).as_quat()])        # Positive 135 degrees rotation in Y axis 
    rot_neg90_y[:, 0], rot_neg90_y[:, 1:] = rot_neg90_y.clone()[:, 3], rot_neg90_y.clone()[:, :3]
    rot_pos135_z[:, 0], rot_pos135_z[:, 1:] = rot_pos135_z.clone()[:, 3], rot_pos135_z.clone()[:, :3]
    
    object_translation, object_rotation = combine_frame_transforms(t01 = object_translation, q01 = rot_neg90_y ,
                                                                   t12 = torch.zeros_like(object_translation), q12 = rot_pos135_z)

    object_translation = object_translation[0].numpy().tolist()
    object_rotation = object_rotation[0].numpy().tolist()

    rot_neg90_y, rot_pos135_z = None, None



    # ---- Reward variables ----
    # reward scales
    rew_scale_dist: float= 1.0

    # Position threshold for ending the episode
    distance_thres = 0.05 # 0.08 # 0.03

    # Bonus for reaching the target
    bonus_tgt_reached = 100




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

    cfg.target_pose = torch.tensor(cfg.target_pose).repeat(num_envs, 1).to(device)
    cfg.target_pose_2 = torch.tensor(cfg.target_pose_2).repeat(num_envs, 1).to(device)
    # cfg.object_base_pose = torch.tensor(cfg.object_base_pose).repeat(num_envs, 1).to(device)
    # cfg.object_increments = torch.tensor(cfg.object_increments).repeat(num_envs, 1).to(device)
    cfg.moving_joints_gripper = torch.tensor(cfg.moving_joints_gripper).repeat(num_envs, 1).to(device)
    cfg.camera_trans = torch.tensor(cfg.camera_trans).repeat(num_envs, 1).to(device)
    cfg.camera_rot = torch.tensor(cfg.camera_rot).repeat(num_envs, 1).to(device)
    cfg.camera_ext_trans = torch.tensor(cfg.camera_ext_trans).repeat(num_envs, 1).to(device)
    cfg.camera_ext_rot = torch.tensor(cfg.camera_ext_rot).repeat(num_envs, 1).to(device)
    cfg.camera_ext_trans_front = torch.tensor(cfg.camera_ext_trans_front).repeat(num_envs, 1).to(device)
    cfg.camera_ext_rot_front = torch.tensor(cfg.camera_ext_rot_front).repeat(num_envs, 1).to(device)
    cfg.ee_translation = torch.tensor(cfg.ee_translation).repeat(num_envs, 1).to(device)
    cfg.ee_rotation = torch.tensor(cfg.ee_rotation).repeat(num_envs, 1).to(device)
    cfg.object_translation = torch.tensor(cfg.object_translation).repeat(num_envs, 1).to(device)
    cfg.object_rotation = torch.tensor(cfg.object_rotation).repeat(num_envs, 1).to(device)
    cfg.contact_matrix = cfg.contact_matrix.repeat(num_envs, 1).to(device)
    cfg.rot_neg90_xy = torch.tensor(cfg.rot_neg90_xy).repeat(num_envs, 1).to(device)
    cfg.rot_neg90_xy_2 = torch.tensor(cfg.rot_neg90_xy_2).repeat(num_envs, 1).to(device)
    cfg.rot_neg90_xy_3 = torch.tensor(cfg.rot_neg90_xy_3).repeat(num_envs, 1).to(device)

    cfg.rot_45_z_pos_quat = torch.tensor(cfg.rot_45_z_pos_quat).repeat(num_envs, 1).to(device)
    cfg.rot_180_z_pos_quat = torch.tensor(cfg.rot_180_z_pos_quat).repeat(num_envs, 1).to(device)

    cfg.obst_list = torch.tensor(cfg.obst_list).repeat(num_envs, 1).to(device)

    return cfg



# Add the collision sensors to the configuration class according to the number of environments
def update_collisions(cfg, num_envs):


    finger_middle_w_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.robot] + "/robotiq_finger_middle_.*",
        update_period=0.001, 
        history_length=1, 
        debug_vis=False,
        filter_prim_paths_expr = [f"/World/envs/env_{i}/object" for i in range(num_envs)],
    )

    finger_1_w_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.robot] + "/robotiq_finger_1_.*",
        update_period=0.001, 
        history_length=1, 
        debug_vis=False,
        filter_prim_paths_expr = [f"/World/envs/env_{i}/object" for i in range(num_envs)],
    )

    finger_2_w_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.robot] + "/robotiq_finger_2_.*",
        update_period=0.001, 
        history_length=1, 
        debug_vis=False,
        filter_prim_paths_expr = [f"/World/envs/env_{i}/object" for i in range(num_envs)],
    )

    robot_w_shelf: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.robot] + "/robotiq_.*",
        update_period=0.001, 
        history_length=1, 
        debug_vis=False,
        filter_prim_paths_expr = [f"/World/envs/env_{i}/shelf" for i in range(num_envs)],
    )



    # Dictionary of contact sensors configurations
    cfg.contact_sensors_dict = {
                                "finger_middle_w_object": finger_middle_w_object,
                                "finger_1_w_object": finger_1_w_object,
                                "finger_2_w_object": finger_2_w_object,
                                "robot_w_shelf": robot_w_shelf ,
                                }
    
    # Updated contact matrix
    cfg.contact_matrix = torch.tensor([2.5, 2.5, 2.5, -10])
    cfg.contact_thres = 6

    return cfg