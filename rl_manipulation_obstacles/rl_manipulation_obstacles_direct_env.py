from __future__ import annotations

import os
import torch
from collections.abc import Sequence
import copy

from .rl_manipulation_obstacles_direct_env_cfg import RLManipulationObstaclesDirectCfg, update_cfg, update_collisions

from .py_dq.src.dq import *
from .py_dq.src.distances import *
from .py_dq.src.dq_lie import *
from .py_dq.src.interpolators import *

from .py_dq.src.quat_trans_lie import *
from .py_dq.src.matrix_lie import *
from .py_dq.src.euler import *

from .train.sam2.networks_lfd import *
from .train.sim_utils import *

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import TiledCamera, ContactSensor

from .train.sam2.data import *

import numpy as np
import matplotlib.pyplot as plt

# from pynput import keyboard

import sys
sys.path.append('../../../')

from .mpc_controller import drop_MPC_setup, save_traj



def save_images_grid(
    images: list[torch.Tensor],
    cmap: str | None = None,
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save images in a grid with optional subtitles and title.

    Args:
        images: A list of images to be plotted. Shape of each image should be (H, W, C).
        cmap: Colormap to be used for plotting. Defaults to None, in which case the default colormap is used.
        nrows: Number of rows in the grid. Defaults to 1.
        subtitles: A list of subtitles for each image. Defaults to None, in which case no subtitles are shown.
        title: Title of the grid. Defaults to None, in which case no title is shown.
        filename: Path to save the figure. Defaults to None, in which case the figure is not saved.
    """
    
    # Show images in a grid
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2)) # -> tiene que ser un numpy array
    axes = np.array(axes)

    # Axes(0.125,0.11;0.775x0.77)
    axes = axes.flatten()

    # Plot images
    for idx, (img, ax) in enumerate(zip(images, axes)):
        img = img.detach().cpu().numpy()
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        # if subtitles:
        #     ax.set_title(subtitles[idx])

    # Remove extra axes if any
    for ax in axes[n_images:]:
        fig.delaxes(ax)

    # # Set title
    # if title:
    #     plt.suptitle(title)

    # Adjust layout to fit the title
    # plt.tight_layout()

    # Save the figure
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)

    # Close the figure
    plt.close()


def get_frame(x, tgt, obst_centers=None, obst_radii=None, traj=None, ax=None, rot = False):
    """
    x     : current state (at least X,Y,Z)
    tgt   : [Xt, Yt, Zt]
    obst_centers : list or array [[xo, yo, zo], ...]
    obst_radii   : list or array [[rx, ry, rz], ...]
    traj  : list or array of past positions [[x,y,z], ...]
    ax    : matplotlib 3D axis
    """

    

    def plot_ellipsoid(ax, center, radii, color='orange', alpha=1, resolution=20):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)

        x = radii[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2]

        ax.plot_surface(
            x, y, z,
            color=color,
            alpha=alpha,
            linewidth=0,
            shade=True,
            zorder = 6
        )

    # =========================
    # Extract positions
    # =========================
    X, Y, Z = float(x[0 + 3*int(rot)]), float(x[1 + 3*int(rot)]), float(x[2 + 3*int(rot)])
    Xt, Yt, Zt = float(tgt[0 + 3*int(rot)]), float(tgt[1 + 3*int(rot)]), float(tgt[2 + 3*int(rot)])

    # =========================
    # Create / clear axis
    # =========================
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()
        ax.cla()

    # =========================
    # Plot trajectory
    # =========================
    if traj is not None and len(traj) > 1:
        traj = np.asarray(traj)
        ax.plot(
            traj[:, 0 + 3*int(rot)], traj[:, 1 + 3*int(rot)], traj[:, 2 + 3*int(rot)],
            'k-', linewidth=2, alpha=0.7, label="Trajectory"
        )
    # =========================
    # Plot obstacles (ellipsoids)
    # =========================
    if obst_centers.cpu().numpy().tolist()[0] != [] and obst_radii.cpu().numpy().tolist()[0] != []:
        for center, radii in zip(obst_centers, obst_radii):
            plot_ellipsoid(ax, center, radii)

    # =========================
    # Plot robot
    # =========================
    ax.scatter(X, Y, Z, c='k', s=80, label="Robot", zorder=5)

    # =========================
    # Plot target
    # =========================
    ax.scatter(Xt, Yt, Zt, c='g', s=100, label="Target", zorder=5)


    # =========================
    # Plot origin
    # =========================
    ax.scatter(0, 0, 0, c='r', s=80, label="Origin")

    # =========================
    # Axis limits
    # =========================
    xs = [X, Xt, 0]
    ys = [Y, Yt, 0]
    zs = [Z, Zt, 0]


    if obst_centers.cpu().numpy().tolist()[0] != []:
        for c in obst_centers:
            xs.append(c[0])
            ys.append(c[1])
            zs.append(c[2])

    margin = 0.5
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_zlim(min(zs) - margin, max(zs) + margin)

    # =========================
    # Formatting
    # =========================
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Robot Trajectory with Obstacles")
    ax.legend(loc="upper left")



    return fig, ax



# Class for the Bimanual Direct Environment
class RLManipulationObstaclesDirect(DirectRLEnv):
    cfg: RLManipulationObstaclesDirectCfg

    # --- init function ---
    def __init__(self, cfg: RLManipulationObstaclesDirectCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # --- Debug variables ---
        # Debug poses for the object and end effector of the GEN3 robot. These poses 
        # are used to draw the markers in the simulation
        self.debug_robot_ee_pose_w = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)
        self.debug_target_pose_w = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)

        # Poses for the object and GEN3 robot so they can match when performing the grasping
        self.target_pose_r =  torch.tensor([0.0 ,0.0 ,0.0, 1.0 ,0.0 ,0.0 ,0.0]).to(self.device).repeat(self.num_envs, 1).float()
        self.target_pose_r_group =  torch.zeros((self.num_envs, cfg.size_group)).to(self.device).float()
        self.target_pose_r_lie = torch.zeros((self.num_envs, cfg.size)).to(self.device).float()

        self.interm_pose_r =  torch.tensor([0.0 ,0.0 ,0.0, 1.0 ,0.0 ,0.0 ,0.0]).to(self.device).repeat(self.num_envs, 1).float()
        self.interm_pose_r_group =  torch.zeros((self.num_envs, cfg.size_group)).to(self.device).float()
        self.interm_pose_r_lie = torch.zeros((self.num_envs, cfg.size)).to(self.device).float()



        self.gripper_pose_r =  torch.tensor([0.0 ,0.0 ,0.0, 1.0 ,0.0 ,0.0 ,0.0]).to(self.device).repeat(self.num_envs, 1).float()
        self.gripper_pose_r_group =  torch.zeros((self.num_envs, cfg.size_group)).to(self.device).float()
        self.gripper_pose_r_lie = torch.zeros((self.num_envs, cfg.size)).to(self.device).float()



        self.object_pose_w_lab = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).repeat(self.num_envs, 1).to(self.device)
        
        self.robot_rot_ee_pose_r_lie_rel = torch.zeros((self.num_envs, self.cfg.size)).to(self.device).float()
        self.robot_rot_ee_pose_r_lie = torch.zeros((self.num_envs, self.cfg.size)).to(self.device).float()

        self.root_robot_pose = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.root_state_w[:, 0:7]
        

        # Indexes for: robot joints, hand joints, all joints
        self._robot_joints_idx = self.scene.articulations[self.cfg.keys[self.cfg.robot]].find_joints(self.cfg.joints[self.cfg.robot])[0]
        self._hand_joints_idx = self.scene.articulations[self.cfg.keys[self.cfg.robot]].find_joints(self.cfg.hand_joints[self.cfg.robot])[0]
        self._all_joints_idx = self.scene.articulations[self.cfg.keys[self.cfg.robot]].find_joints(self.cfg.all_joints[self.cfg.robot])[0]

        # IK Controller
        controller_cfg = DifferentialIKControllerCfg(command_type = "pose", use_relative_mode = False, ik_method = "pinv")
        # DifferentialIKControllerCfg: Configuration for differential inverse kinematics controller.
        #    command_type: Type of task-space command to control the articulation's body.
        #    use_relative_mode: Whether to use relative mode for the controller. --> Use increments for the positions.
        #    ik_method: Method for computing inverse of Jacobian.

        self.controller = DifferentialIKController(controller_cfg, num_envs = self.num_envs, device = self.device)
        # DifferentialIKController: Differential inverse kinematics (IK) controller.
        #    num_envs: Number of environments handled by the controller.
        #    device: Device into which the controller is stored.


        # Indexes for the both robots' end effector for Jacobian computation
        self.ee_jacobi_idx = self.scene.articulations[self.cfg.keys[self.cfg.robot]].find_bodies(self.cfg.ee_link[self.cfg.robot])[0][0] - 1
        self.camera_idx = self.scene.articulations[self.cfg.keys[self.cfg.robot]].find_bodies("wrist_3_link")[0][0]
       

        # List for the default joint poses of both robots --> As a list due to the different joints of the arms (6 and 7) 
        self.default_joint_pos = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.default_joint_pos


        # List of joint actions
        self.actions = torch.zeros((self.num_envs, len(self._all_joints_idx))).to(self.device)

        # Poses obtained at reset
        self.reset_robot_poses_r = torch.zeros((self.num_envs, 7)).to(self.device)

        # Update configuration class
        self.cfg = update_cfg(cfg = cfg, num_envs = self.num_envs, device = self.device)

        # Obtain the number of contact sensors per environment
        self.num_contacts = 0
        for __ in self.cfg.contact_sensors_dict:
            self.num_contacts += 1

        # Variable to store contacts between prims
        self.contacts = torch.empty(self.num_envs, self.num_contacts).fill_(False).to(self.device)
        self.contacts_weight = torch.zeros(self.num_envs, self.num_contacts).to(self.device)
        self.contacts_w = torch.empty(self.num_envs).fill_(False).to(self.device)
        self.is_contact = torch.empty(self.num_envs).fill_(False).bool().to(self.device)

        self.z_displ = torch.tensor([0.0, -0.0, 0.03]).to(self.device).repeat(self.num_envs, 1) # -0.21


        # Obtain the ranges in which sample reset poses
        self.ee_pose_ranges = torch.tensor([[ [(i + cfg.apply_range[idx]*inc[0]), (i + cfg.apply_range[idx]*inc[1])] for i, inc in zip(poses, cfg.ee_pose_incs)] for idx, poses in enumerate(cfg.ee_init_pose)]).to(self.device)
        self.target_pose_ranges = torch.tensor([[ [(i + cfg.apply_range_tgt*inc[0]), (i + cfg.apply_range_tgt*inc[1])] for i, inc in zip(poses, cfg.target_poses_incs)] for poses in cfg.target_pose]).to(self.device)
        self.target_pose_ranges2 = torch.tensor([[ [(i + cfg.apply_range_tgt*inc[0]), (i + cfg.apply_range_tgt*inc[1])] for i, inc in zip(poses, cfg.target_poses_incs2)] for poses in cfg.target_pose_2]).to(self.device)
        
        # Previous distance
        self.prev_dist = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)

        # Target reached flag
        self.target_reached = torch.zeros(self.num_envs).to(self.device).bool()
        self.home_reached = torch.zeros(self.num_envs).to(self.device).bool()

        # --- Lie algebra ---
        # List of mappings
        map_list = [[[identity_map, identity_map], [exp_bruno, log_bruno],     [exp_stereo, log_stereo]],
                    [[identity_map, identity_map],],
                    [[identity_map, identity_map], [exp_quat_stereo, log_quat_stereo]],
                    [[identity_map, identity_map], [exp_se3, log_se3]]]

        # List of conversions
        conversions = [[convert_dq_to_Lab, dq_from_tr], 
                       [convert_euler_to_Lab, from_quat_to_euler], 
                       [convert_quat_trans_to_Lab, identity_map_conversion], 
                       [convert_homo_to_Lab, homo_from_mat_trans_LAB]]
        
        diff_operators = [dq_diff, euler_diff, q_trans_diff, mat_diff]
        mul_operators = [dq_mul, euler_mul, q_trans_mul, mat_mul]

        # List of interpolators
        interpolators = [ScLERP, None, None, None]

        # Lis of distance functions
        distances = [[dqLOAM_distance, geodesic_dist, double_geodesic_dist],
                     [geodesic_dist],
                     [geodesic_dist],
                     [geodesic_dist]]
        
        identities = [[1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                      [0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0,   1.0, 0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0, 0.0,   0.0, 1.0, 0.0, 0.0,   0.0, 0.0, 1.0, 0.0,   0.0, 0.0, 0.0, 1.0]]
        
        normalizes = [dq_normalize, euler_normalize, norm_quat, norm_mat]

        # Assign the functions according to configuration
        self.exp = map_list[cfg.representation][cfg.mapping][0]                 # Exponential mapping
        self.log = map_list[cfg.representation][cfg.mapping][1]                 # Logarithmic mapping
        self.convert_to_Lab = conversions[cfg.representation][0]                # Conversion Lie group to IsaacLab representation
        self.convert_to_group = conversions[cfg.representation][1]              # Conversion IsaacLab representation to Lie group     
        self.interpolator = interpolators[cfg.representation]                   # Interpolator function
        self.dist_function = distances[cfg.representation][cfg.distance]        # Distance function
        self.diff_operator = diff_operators[cfg.representation]
        self.mul_operator = mul_operators[cfg.representation]
        self.normalize = normalizes[cfg.representation]
    
        # Initial pose in the group
        self.pose_group_r = torch.tensor(identities[cfg.representation]).to(self.device).repeat(self.num_envs, 1).float()
        self.reset_robot_poses_group_r = torch.tensor(identities[cfg.representation]).to(self.device).repeat(self.num_envs, 1).float()


        # ----------- AUX --------------        
        self.prim_action = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).repeat(self.num_envs, 1).to(self.device)
        self.correspondences = ['w','s','a','d','o','l']
        self.gripper = ['0', '1']
        self.gripper_inc = 5
        self.prim_g_action = torch.tensor([0.0]).repeat(self.num_envs, 1).to(self.device)
        self.inc = 0.01

        # Crear listener
        # listener = keyboard.Listener(on_press=self.on_press)
        # listener.start()  # ✅ No bloquea
        # --------------------------------


        self.end_target_pose_r = torch.tensor([[-0.4308,  0.1459,  0.4802 - 0.25,  0.1308, -0.4781, -0.8669, -0.0536]], device=self.device).repeat(self.num_envs, 1)
        self.end_target_pose_r_group = self.convert_to_group(self.end_target_pose_r[:, :3], self.end_target_pose_r[:, 3:])
        self.end_target_pose_r_lie = self.log(self.end_target_pose_r_group)


        # --- Camera poses ---
        self.cfg.camera_ext_trans, self.cfg.camera_ext_rot = combine_frame_transforms(t01 = self.root_robot_pose[:, :3],     q01 = self.root_robot_pose[:, 3:7],
                                                                                      t12  =self.cfg.camera_ext_trans,   q12 = self.cfg.camera_ext_rot)
        
        self.cfg.camera_ext_trans_front, self.cfg.camera_ext_rot_front = combine_frame_transforms(t01 = self.root_robot_pose[:, :3],     q01 = self.root_robot_pose[:, 3:7],
                                                                                      t12  =self.cfg.camera_ext_trans_front,   q12 = self.cfg.rot_neg90_xy_2)
        


        self.cfg.camera_ext_trans_front, self.cfg.camera_ext_rot_front = combine_frame_transforms(t01 = self.cfg.camera_ext_trans_front,   q01 = self.cfg.camera_ext_rot_front,
                                                                                                  t12 = torch.zeros_like(self.cfg.camera_ext_trans_front).to(self.device),   q12 = self.cfg.rot_neg90_xy_3)


        # new_ext_pos, new_ext_rot = combine_frame_transforms(t01 =self.cfg.camera_ext_trans,   q01 = self.cfg.camera_ext_rot,
        #                                                     t12 = torch.zeros_like(self.cfg.camera_ext_trans).to(self.device),   q12 = self.cfg.rot_neg90_xy)
        self.scene.sensors["camera_ext"].set_world_poses(positions = self.cfg.camera_ext_trans, orientations = self.cfg.camera_ext_rot)
        self.scene.sensors["camera_front"].set_world_poses(positions = self.cfg.camera_ext_trans_front, orientations = self.cfg.camera_ext_rot_front)

        self.camera_ext = None
        self.camera_w = None
        self.camera_front = None

        self.pc_w = None
        self.pc_ext = None
        self.pc_front = None

        self.u_opt = torch.tensor([[0, 0, 0, 0, 0, 0]]).to(self.device)
        self.x0 = torch.tensor([[0, 0, 0, 0, 0, 0]]).to(self.device)
        self.prev_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).to(self.device)


        self.trajectory_save = []

        self.episode_id = 0

        self.current_path = os.path.dirname(os.path.realpath(__file__))


        # self.my_traj = torch.load("/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/aurova_reinforcement_learning/rl_manipulation_obstacles/traj.pt")
        self.my_dict = None

        # with open("/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/aurova_reinforcement_learning/rl_manipulation_obstacles/traj.pkl", 'rb') as f:
        #     self.m_dict = pickle.load(f)

        # lie = self.my_dict["lie"]
        # self.my_traj = self.my_dict["traj"]
        # if lie:
        #     self.my_traj = self.convert_to_Lab(self.exp(self.my_traj)).to(self.device)
        # else:
        #     new_quat = quat_from_euler_xyz(self.my_traj[:, 3], self.my_traj[:, 4], self.my_traj[:, 5])
        #     self.my_traj = torch.cat((self.my_traj[:, :3], new_quat), dim = -1).to(self.device)


    def on_press(self, key):
        try:
            if key.char in self.correspondences:
                idx = self.correspondences.index(key.char)
                
                prim_idx = int(idx/2)

                prim_inc = (2 * int(idx%2 == 0) - 1)

                self.prim_action = self.prim_action.clone()
                self.prim_action[:, prim_idx] = prim_inc * self.inc


            if key.char in self.gripper:
                inc = 2*int(key.char) - 1

                self.prim_g_action = torch.tensor(inc).repeat(self.num_envs, 1).to(self.device)

            
        except AttributeError:
            pass


    # Method to add all the prims to the scene --> Overrides method of DirectRLEnv
    def _setup_scene(self, ):
        '''
        NOTE: The "self.scene" variable is declared at "super().__init__(cfg, render_mode, **kwargs)" in __init__
        '''
    
        # Add ground plane
        spawn_ground_plane(prim_path = "/World/ground", cfg = GroundPlaneCfg())

        # Clone, filter and replicate
        self.scene.clone_environments(copy_from_source=False)
        # clone_environments: Creates clones of the environment /World/envs/env_0.
        #     if "copy_from_source" is False, clones inherit from /World/envs/env_0 and mirror its changes.

        self.scene.filter_collisions(global_prim_paths=[])
        # filter_collisions: Disables collisions between the environments in /World/envs/env_.* and enables collisions with the prims in global prim paths (e.g. ground plane).
        #     if "global_prim_paths" is None, environments do not collide with each other.

        # Add articulations to scene
        if self.cfg.robot == self.cfg.UR5e:
            self.scene.articulations[self.cfg.keys[self.cfg.robot]] = Articulation(self.cfg.robot_cfg_1)
        
        elif self.cfg.robot == self.cfg.GEN3:
            self.scene.articulations[self.cfg.keys[self.cfg.robot]] = Articulation(self.cfg.robot_cfg_2)

        elif self.cfg.robot == self.cfg.UR5e_3f:
            self.scene.articulations[self.cfg.keys[self.cfg.robot]] = Articulation(self.cfg.robot_cfg_3)

        elif self.cfg.robot == self.cfg.UR5e_NOGRIP:
            self.scene.articulations[self.cfg.keys[self.cfg.robot]] = Articulation(self.cfg.robot_cfg_4)
        
        # self.scene.rigid_objects["shelf"] = RigidObject(self.cfg.shelf_cfg)
        self.scene.rigid_objects["object"] = RigidObject(self.cfg.object_cfg)

        # Add extras (markers, ...)
        self.scene.extras["markers"] = VisualizationMarkers(self.cfg.marker_cfg)

        self.scene.sensors["camera"] = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["camera_ext"] = TiledCamera(self.cfg.tiled_camera_ext)
        self.scene.sensors["camera_front"] = TiledCamera(self.cfg.tiled_camera_front)

        # Correct collision sensors 
        self.cfg = update_collisions(self.cfg, num_envs = self.num_envs)
        for idx, sensor_cfg in self.cfg.contact_sensors_dict.items():
            self.scene.sensors[idx] = ContactSensor(sensor_cfg)


        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    # Method to preprocess the actions so they have a proper format
    def _preprocess_actions(self, actions: torch.Tensor, gripper = True) -> torch.Tensor:
        '''
        In:
            - actions - torch.Tensor (N, m): incremental actions in the Lie algebra

        Out:
            - actions - torch.Tensor: preprocessed actions.
        '''

        # Clamp actions
        if self.cfg.representation == self.cfg.MAT and self.cfg.mapping == 0:
            actions[:, :3] = torch.clamp(actions[:, :3], -self.cfg.action_scaling[0], self.cfg.action_scaling[0])
            actions[:, 4:7] = torch.clamp(actions[:, 4:7], -self.cfg.action_scaling[0], self.cfg.action_scaling[0])
            actions[:, 8:11] = torch.clamp(actions[:, 8:11], -self.cfg.action_scaling[0], self.cfg.action_scaling[0])

            actions[:, 3] = torch.clamp(actions[:,3], -self.cfg.action_scaling[1], self.cfg.action_scaling[1])
            actions[:, 7] = torch.clamp(actions[:, 7], -self.cfg.action_scaling[1], self.cfg.action_scaling[1])
            actions[:, 11] = torch.clamp(actions[:, 11], -self.cfg.action_scaling[1], self.cfg.action_scaling[1])

        else:
            actions[:, :3] = torch.clamp(actions[:, :3], -self.cfg.action_scaling[0], self.cfg.action_scaling[0])
            # actions[:, 3:] = torch.clamp(actions[:, 3:], -self.cfg.action_scaling[1], self.cfg.action_scaling[1])

            if gripper:
                actions[:, 3:-1] = torch.clamp(actions[:, 3:-1], -self.cfg.action_scaling[1], self.cfg.action_scaling[1])
                actions[:, -1] = torch.clamp(actions[:, -1], -self.cfg.grip_scaling, self.cfg.grip_scaling)

            else:
                actions[:, 3:] = torch.clamp(actions[:, 3:], -self.cfg.action_scaling[1], self.cfg.action_scaling[1])


        return actions
    
    
    # Obtain the end effector pose of the robot in the base frame
    def _get_ee_pose(self):
        '''
        In: 
            - None

        Out:
            - ee_pos_r - torch.tensor(N, 3): position of the end effector in the base frame for each environment.
            - ee_quat_r - torch.tensor(N, 4): orientation as a quaternions of the end effector in the base frame for each environment.
            - jacobian - torch.tensor(N, 6, n_joints (6 or 7)): jacobian of all robots' end effector. 
            - joint_pos - torch.tensor(N, n_joints(6 or 7)): joint position of the robot.
        '''
        
        # Obtains the jacobian of the end effector of the robot
        jacobian = self.scene.articulations[self.cfg.keys[self.cfg.robot]].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self._robot_joints_idx]

        # Obtains the pose of the end effector in the world frame
        ee_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.body_state_w[:, self.ee_jacobi_idx+1, 0:7]

        # Obtains the pose of the base of the robot in the world frame
        root_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.root_state_w[:, 0:7]
        
        # Obtains the joint position
        joint_pos = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.joint_pos[:, self._robot_joints_idx]

        # Transforms end effector frame coordinates (in world) into root (local / base) coordinates
        ee_pos_r, ee_quat_r = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
        # root = T01 // ee = T02 -> substract = (T01)^-1 * T02 = T10 * T02 = T12
        
        return ee_pos_r, ee_quat_r, jacobian, joint_pos


    # Performs the action increment
    def perform_increment(self, actions):
        '''
        In: 
            - actions - torch.tensor(N, m): the increment to be performed to the actual pose in the Lie algebra.

        Out:
            - None
        '''

        grip_action = actions[:, -1]
        # actions = actions[:, :-1]

        # # Perform increment in the algebra and exponential map -> (plus operator)
        # action_pose = self.exp(self.robot_rot_ee_pose_r_lie_rel + actions)
        # action_pose = self.mul_operator(self.target_pose_r_group, action_pose)
        # action_pose = self.normalize(action_pose)

        # # Convert to IsaacLab representation (translation, quaternion)
        # action_pose_lab = self.convert_to_Lab(action_pose)


        
        if not self.cfg.test:
            if self.count < self.subs_limit:
                self.trajectory_save[self.count][:3] = self.target_pose_r_lie[:, :3]
            else:
                self.trajectory_save[self.count][:3] = self.end_target_pose_r_lie[:,:3]
            cmd_lie = self.trajectory_save[self.count].repeat(self.num_envs, 1)
            cmd = self.convert_to_Lab(self.exp(cmd_lie))
            
            cmd = combine_frame_transforms(t01= cmd[:, :3],  q01 = cmd[:, 3:],
                                        t12 = -self.cfg.ee_translation,   q12 = self.cfg.ee_rotation) 
                                           


            self.gripper_action = self.count >= self.start_grip_idx and (not self.is_contact.item())
            
            self.increment_condition = (self.count < self.trajectory_save.shape[0] and not self.gripper_action) or self.increment_condition

            if self.increment_condition:
                self.count += 1

            grip_action += self.cfg.grip_scaling * int(self.gripper_action) 
        
        else:
            
            cmd = self.test_model(self.camera_w[-1].repeat(3,1,1).unsqueeze(0) / 255.0, 
                                  self.camera_ext[-1].repeat(3,1,1).unsqueeze(0) / 255.0,
                                  self.camera_front[-1].repeat(3,1,1).unsqueeze(0) / 255.0,
                                  self.gripper_pose_r_lie)
            
            # actions = self._preprocess_actions(cmd)

            grip_action = cmd[:, -1].clone()
            cmd_lie = cmd[:, :-1].clone()

            cmd = self.convert_to_Lab(self.exp(cmd_lie))
            
            cmd = combine_frame_transforms(t01= cmd[:, :3],                  q01 = cmd[:, 3:],
                                           t12 = -self.cfg.ee_translation,   q12 = self.cfg.ee_rotation)
            
            self.count += 1
            

        self.controller.set_command(torch.cat(cmd, dim = -1))

        # Set the command for the IKDifferentialController
        
        # Obtains the poses
        ee_pos_r, ee_quat_r, jacobian, joint_pos = self._get_ee_pose()

        
        # Get the actions for the robot. Concatenates:
        #   - the joint coordinates for the action computed by the IKDifferentialController and
        #   - the joint coordinates for the hand.
        self.actions[:, :6] = self.controller.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos)


        # --- Update gripper position ---
        actual_gripper_pos = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.joint_pos[:, self._hand_joints_idx]
        
        # move_hand = (self.hand_pose*140) < 125.0

        self.actions[:, 6:] = grip_action.unsqueeze(-1) * self.cfg.moving_joints_gripper + actual_gripper_pos


    # Method called before executing control actions on the simulation --> Overrides method of DirecRLEnv
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        '''
        In: 
            - actions - torch.tensor(N, ...): actions to apply to the environment.

        Out:
            - None
        '''


        # Preprocessing actions
        # actions = self._preprocess_actions(actions)

        # Obtains the increments and the poses
        self.perform_increment(actions = actions)

        if not self.cfg.test and self.count % self.cfg.save_interval == 0 and self.count < (len(self.trajectory_save) - self.cfg.save_interval):
            self.save_step()


    # Applies the preprocessed action in the environment --> Overrides method of DirecRLEnv
    def _apply_action(self) -> None:

        # Applies joint actions to the robot
        self.scene.articulations[self.cfg.keys[self.cfg.robot]].set_joint_position_target(self.actions, joint_ids=self._all_joints_idx)


    # Update the position of the markers with debug purposes
    def update_markers(self):
        '''
        Current markers:
            - End effector of the robot.
            - Pose of the object.
        '''

        # Obtains a tensor of indices (a tensor containing tensors from 0 to the number of markers)
        marker_indices = torch.arange(self.scene.extras["markers"].num_prototypes).repeat(self.num_envs)

        # Updates poses in simulation
        self.scene.extras["markers"].visualize(translations = torch.cat((self.target_pose_r[:, :3], 
                                                                         self.gripper_pose_r[:, :3],
                                                                         self.interm_pose_r[:, :3])), 
                                                                         
                                                orientations = torch.cat((self.target_pose_r[:, 3:], 
                                                                          self.gripper_pose_r[:,3:],
                                                                          self.interm_pose_r[:, 3:])), 

                                                marker_indices=marker_indices)


    # Method to filter collisions according to the force matrix
    def filter_collisions(self):
        '''
        In:
            - None
        
        Out:
            - None
        '''

        # Loop through all the contact sensors configuration for the indexes
        for idx, (key, __) in enumerate(self.cfg.contact_sensors_dict.items()):

            # Obtain the matrix -> reshape it -> sum the last two dimensions -> 
            #    -> if the value is greater than 0, there is force so  there is contact
            self.contacts[:, idx] = torch.abs(self.scene.sensors[key].data.force_matrix_w).view(self.num_envs, -1, 3).sum(dim = (1,2), keepdim = True).squeeze((-2, -1)) > 0.0

        self.contacts_weight = (self.contacts * self.cfg.contact_matrix)
        self.contacts_w = self.contacts_weight[:, :-1].sum(-1)
        self.is_contact = (self.contacts_w > self.cfg.contact_thres)


    # Updates the poses of the object and robot so they can match when performing the grasp
    def update_new_poses(self):
        '''
        In:
            - None
        
        Out:
            - None
        '''

        # --- Robot poses ---
        # Obtain the pose of the UR5e end effector in world frame
        self.debug_robot_ee_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.body_state_w[:, self.ee_jacobi_idx+1, 0:7]

        # Obtain the pose of the end effector in UR5e root frame
        robot_rot_ee_pos_r, robot_rot_ee_quat_r = subtract_frame_transforms(t01 = self.root_robot_pose[:, :3],         q01 = self.root_robot_pose[:, 3:],
                                                                            t02 = self.debug_robot_ee_pose_w[:, :3],   q02 = self.debug_robot_ee_pose_w[:, 3:])

        # Fix double cover
        neg_idx = robot_rot_ee_quat_r[:, 0] < 0.0
        robot_rot_ee_quat_r[neg_idx] *= -1
        


        # --- Target pose ---
        # Get object world pose
        self.debug_target_pose_w = self.scene.rigid_objects["object"].data.body_state_w[:, 0, 0:7]


        # alpha = atan((self.debug_target_pose_w[0 ,1].item() - self.debug_robot_ee_pose_w[0, 1].item()) / (self.debug_target_pose_w[0 ,0].item() - self.debug_robot_ee_pose_w[0, 0].item()))
        # rot_negAlpha_yz = torch.tensor([(Rotation.from_rotvec((alpha) * np.array([0, 0, 1]))).as_quat()]).to(self.device)
        # w = rot_negAlpha_yz[:, 3].clone().unsqueeze(0)
        # xyz = rot_negAlpha_yz[:, :3].clone()
        # rot_negAlpha_yz = torch.cat((w,xyz), dim=-1)



        # corrected_ref = combine_frame_transforms(t01 = self.debug_target_pose_w[:, :3], q01 = self.debug_target_pose_w[:, 3:],
        #                                         t12 = torch.zeros(1,3).to(self.device), q12 = rot_negAlpha_yz)
        # self.debug_target_pose_w = torch.cat(corrected_ref ,dim = -1).float()




        # Obtain the relative pose w.r.t. the robot root frame
        target_pos_r, target_quat_r = subtract_frame_transforms(t01 = self.root_robot_pose[:, :3], q01 = self.root_robot_pose[:, 3:],
                                                                t02 = self.debug_target_pose_w[:, :3], q02 = self.debug_target_pose_w[:, 3:])
        
        target_pos_r, target_quat_r = combine_frame_transforms(t01 = target_pos_r, q01 = target_quat_r,
                                                                     t12 = self.z_displ, q12 = self.cfg.rot_45_z_pos_quat)
        
        target_pos_w, target_quat_w = combine_frame_transforms(t01 = self.root_robot_pose[:, :3],        q01 = self.root_robot_pose[:, 3:],
                                                               t12 = target_pos_r,                       q12 = target_quat_r)
        
        self.debug_target_pose_w = torch.cat((target_pos_w, target_quat_w), dim = -1)


        # Fix double cover
        # neg_idx = target_quat_r[:, 0] < 0.0
        # target_quat_r[neg_idx] *= -1

        self.target_pose_r = torch.cat((target_pos_r, target_quat_r), dim = -1)
        self.target_pose_r_group = self.convert_to_group(target_pos_r, target_quat_r)
        self.target_pose_r_lie = self.log(self.target_pose_r_group)

        self.interm_pose_r = self.target_pose_r.clone()
        self.interm_pose_r[:, 2] += 0.225
        self.interm_pose_r_group = self.convert_to_group(self.interm_pose_r[:, :3], self.interm_pose_r[:, 3:])
        self.interm_pose_r_lie = self.log(self.interm_pose_r_group)


        # --- Build relative pose observation ---
        # Build the group object
        self.pose_group_r = self.convert_to_group(robot_rot_ee_pos_r, robot_rot_ee_quat_r)

        # Transform to the Lie algebra
        self.robot_rot_ee_pose_r_lie = self.log(self.pose_group_r)
        diff = self.diff_operator(self.target_pose_r_group, self.pose_group_r)
        self.robot_rot_ee_pose_r_lie_rel = self.log(diff)


        

        # Hand observations
        self.hand_joints_pos = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.joint_pos[:, self._hand_joints_idx]
        self.hand_pose = torch.round(self.hand_joints_pos[:, 2] / self.cfg.m[0], decimals = 0) / 140.0



        # Put actual end effector pose on the gripper
        robot_rot_ee_pos_r, robot_rot_ee_quat_r = combine_frame_transforms(t01 = robot_rot_ee_pos_r,       q01 = robot_rot_ee_quat_r,
                                                                           t12 = self.cfg.ee_translation,  q12 = self.cfg.ee_rotation)
        
        self.gripper_pose_r = torch.cat((robot_rot_ee_pos_r, robot_rot_ee_quat_r), dim = -1) 
        self.gripper_group_r = self.convert_to_group(robot_rot_ee_pos_r, robot_rot_ee_quat_r)
        self.gripper_pose_r_lie = self.log(self.gripper_group_r)     


    # Processes the camera poses and images from them
    def _get_images(self, camera_key = "camera"):
        '''
        In:
            - camera_key - str: key of the camera to obtain images from.
        
        Out:
            - imgs - torch.Tensor: image obtained from the desired camera.
        '''

        # self.count += 1
        output_dir = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/aurova_reinforcement_learning/"

        camera_pose = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.body_state_w[:, self.camera_idx, 0:7]
        
        new_camera_trans, new_camera_rot = combine_frame_transforms(t01 = camera_pose[:, :3],     q01 = camera_pose[:, 3:7],
                                                                    t12 = self.cfg.camera_trans,  q12 = self.cfg.camera_rot)
        
        self.scene.sensors[camera_key].set_world_poses(positions = new_camera_trans, orientations = new_camera_rot)
        
        # ---- Get data ----
        cam = self.scene.sensors["camera"].data.output["rgb"][0, ..., :3].permute(2, 0, 1)
        cam_ext = self.scene.sensors["camera_ext"].data.output["rgb"][0, ..., :3].permute(2, 0, 1)
        cam_front = self.scene.sensors["camera_front"].data.output["rgb"][0, ..., :3].permute(2, 0, 1)

        cam_D = self.scene.sensors["camera"].data.output["depth"][0, ..., 0].unsqueeze(0)
        cam_ext_D = self.scene.sensors["camera_ext"].data.output["depth"][0, ..., 0].unsqueeze(0)
        cam_front_D = self.scene.sensors["camera_front"].data.output["depth"][0, ..., 0].unsqueeze(0)

        # cam_ext_D = torch.clip(cam_ext_D, 0.8, 1.0)
        # cam_ext_D = (cam_ext_D - 0.0) / 1.0

        # cam_front_D = torch.clip(cam_front_D, 0.8, 1.0)
        # cam_front_D = (cam_front_D - 0.0) / 1.0


        cam = cam
        cam_ext = cam_ext
        cam_front = cam_front

        cam_D = cam_D
        cam_ext_D = cam_ext_D
        cam_front_D = cam_front_D

        self.camera_w = torch.cat((cam, cam_D), dim = 0)*255
        self.camera_ext = torch.cat((cam_ext, cam_ext_D), dim = 0)*255
        self.camera_front = torch.cat((cam_front, cam_front_D), dim = 0)*255


        # Render images every certain amount of steps
        if self.cfg.save_imgs:
            if self.count % 5 == 0:
            # Function to save images (in utils)
                save_images_grid(images = [self.camera_front[-1]],
                                 subtitles = ["Camera"],
                                 title = "RGB Image: Cam0",
                                 filename = os.path.join(output_dir, "rgb", f"{self.count:04d}.jpg"))
                    

    def save_step(self):
        
        cam = self.camera_w.cpu().numpy().astype(np.uint8)
        cam_ext = self.camera_ext.cpu().numpy().astype(np.uint8)
        cam_front = self.camera_front.cpu().numpy().astype(np.uint8)


        target_pose = self.target_pose_r_lie[0].float().cpu().numpy()
        gripper_pose = self.gripper_pose_r_lie[0].float().cpu().numpy()
        action = self.trajectory_save[self.count].float().cpu().numpy()

        diff = (self.gripper_pose_r_lie - self.prev_pose)[0].float().cpu().numpy()

        pc_w = self.pc_w.float().cpu().numpy() # * 100
        pc_ext = self.pc_ext.float().cpu().numpy() # * 100
        pc_front = self.pc_front.float().cpu().numpy() # * 100

        cam_p = torch.rand((64*64)).float().cpu().numpy()

        # ---- Save step ----
        self.writer.add_step(cam, cam_ext, cam_front, 
                             cam_p, cam_p, cam_p,
                             pc_w, pc_ext, pc_front, 
                             target_pose, gripper_pose, action, diff, self.gripper_action)

        self.prev_pose = self.gripper_pose_r_lie


    def _get_PC(self):
        intrinsics = self.scene.sensors["camera"].data.intrinsic_matrices[0]
        intrinsics_ext = self.scene.sensors["camera_ext"].data.intrinsic_matrices[0]
        intrinsics_front = self.scene.sensors["camera_front"].data.intrinsic_matrices[0]

        pc_w = depth_to_pointcloud(self.camera_w[-1], intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
        pc_ext = depth_to_pointcloud(self.camera_ext[-1], intrinsics_ext[0, 0], intrinsics_ext[1, 1], intrinsics_ext[0, 2], intrinsics_ext[1, 2])
        pc_front = depth_to_pointcloud(self.camera_front[-1], intrinsics_front[0, 0], intrinsics_front[1, 1], intrinsics_front[0, 2], intrinsics_front[1, 2])


        camera_w_pos = self.scene.sensors["camera"].data.pos_w
        camera_w_quat = self.scene.sensors["camera"].data.quat_w_world
        self.pc_w = transform_points(pc_w, camera_w_pos.squeeze(0), camera_w_quat.squeeze(0))

        camera_ext_pos = self.scene.sensors["camera_ext"].data.pos_w
        camera_ext_quat = self.scene.sensors["camera_ext"].data.quat_w_world
        self.pc_ext = transform_points(pc_ext, camera_ext_pos.squeeze(0), camera_ext_quat.squeeze(0))

        camera_front_pos = self.scene.sensors["camera_front"].data.pos_w
        camera_front_quat = self.scene.sensors["camera_front"].data.quat_w_world
        self.pc_front = transform_points(pc_front, camera_front_pos.squeeze(0), camera_front_quat.squeeze(0))


    # Getter for the observations of the environment --> Overrides method of DirectRLEnv
    def _get_observations(self) -> dict:
        '''
        In:
            - None
        
        Out:
            - observations - dict: observations from the environment --> Needs to be with "policy" key. 
        '''

        # Updates the poses of the GEN3 end effector and the object so they match
        self.update_new_poses()
        self.filter_collisions()
        
        self._get_images()
        self._get_PC()
        

        # Builds the tensor with all the observations in a single row tensor (N, 6+1+3+80*80)
        # obs = torch.cat((self.robot_rot_ee_pose_r_lie_rel, self.hand_pose.unsqueeze(-1), self.contacts[:, :3], image), dim = -1)
        obs = torch.cat((self.robot_rot_ee_pose_r_lie_rel, self.hand_pose.unsqueeze(-1), self.contacts[:, :3]), dim = -1)

        # Builds the dictionary
        observations = {"policy": obs}
        
        # Updates markers
        if self.cfg.debug_markers:
            self.update_markers()

        return observations


    # Computes the reward of the transition --> Overrides method of DirectRLEnv
    def _get_rewards(self) -> torch.Tensor:
        '''
        In:
            - None

        Out:
            - compute_rewards() - torch.tensor(N,1): reward for each environment.
        '''  

        # ---- Distance computation ----
        dist_target = self.dist_function(self.pose_group_r, self.target_pose_r_group, self.log, self.diff_operator)
        dist_home = self.dist_function(self.target_pose_r_group, self.reset_robot_poses_group_r, self.log, self.diff_operator)  

        dist = dist_target * torch.logical_not(self.target_reached) + dist_home * self.target_reached                                                                 

        # Obtains wether the agent is approaching or not
        mod = (2*(dist < self.prev_dist).int() - 1).float()

        aux_tgt_reached = self.target_reached.clone()

        # Target reached flag
        self.target_reached = torch.logical_or(torch.logical_and(dist_target < self.cfg.distance_thres, self.is_contact), self.target_reached)
        self.home_reached = dist_home < self.cfg.distance_thres

        apply_bonus = torch.logical_and(self.target_reached, torch.logical_not(aux_tgt_reached))


        # ---- Distance reward ----
        # Reward for the approaching
        reward = mod * self.cfg.rew_scale_dist * torch.exp(-2*dist) + self.contacts_w

        # ---- Reward composition ----
        # Phase reward plus bonuses
        reward = reward +  torch.logical_or(apply_bonus, self.home_reached) * self.cfg.bonus_tgt_reached

        # Update previous distances
        self.prev_dist = dist
        
        return reward
    

    # Verifies when to reset the environment --> Overrides method of DirecRLEnv
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        In:
            - None

        Out:
            - truncated - torch.tensor(N, 1): tensor of boolean indicating if the episodes was truncated (finished badly).
            - terminated - torch.tensor(N, 1): tensor of boolean indicating if the episodes was terminated (finished).
        '''
    
        # Computes time out indicators
        if not self.cfg.test:
            time_out = torch.tensor(self.count >= self.trajectory_save.shape[0] - 1).bool().to(self.device)  # self.episode_length_buf >= self.max_episode_length - 1
            if time_out.item():
                self.writer.close()
        else:
            time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Checks out of bounds in velocity
        out_of_bounds = torch.norm(self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.body_state_w[:, self.ee_jacobi_idx+1, 7:], dim = -1) > self.cfg.velocity_limit 

        # Truncated and terminated variables
        truncated = out_of_bounds
        terminated = torch.logical_or(time_out, self.home_reached)

        return truncated, terminated
    

    # Resets the robot JOINT positions
    def reset_robot(self, env_ids):
        '''
        In:
            - env_ids - torch.tensor(m): IDs for the 'm' environments that need to be resetted.
        
        Out:
            - None
        '''

        # Default joint position for the robots
        joint_pos = self.default_joint_pos[env_ids]
        joint_vel = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.default_joint_vel[env_ids]

        # Write the joint positions to the environments
        self.scene.articulations[self.cfg.keys[self.cfg.robot]].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


    # Resets the robot according to their END EFFECTOR
    def reset_robot_ee(self, env_ids):
        '''
        In:
            - env_ids - torch.tensor(m): IDs for the 'm' environments that need to be resetted.
        
        Out:
            - None
        '''

        # Sample a random position using the end effector ranges with the shape of all environmnets
        ee_init_pose = sample_uniform(
            self.ee_pose_ranges[self.cfg.robot, :, 0],
            self.ee_pose_ranges[self.cfg.robot, :, 1],
            [self.num_envs, self.ee_pose_ranges[0, :, 0].shape[0]],
            self.device,
        )

        # Transforms Euler to quaternion
        quat = quat_from_euler_xyz(roll = ee_init_pose[:, 3],
                                    pitch = ee_init_pose[:, 4],
                                    yaw = ee_init_pose[:, 5])
        
        # Builds the new initial pose
        ee_init_pose = torch.cat((ee_init_pose[:, :3], quat), dim = -1)

        # Save sampled pose
        self.reset_robot_poses_r[env_ids] = ee_init_pose[env_ids]
        self.reset_robot_poses_group_r[env_ids] = self.convert_to_group(ee_init_pose[:, :3], ee_init_pose[:, 3:])

        # Sets the command to the DifferentialIKController
        self.controller.set_command(ee_init_pose)

        # Obtains current poses for the robot
        ee_pos_r, ee_quat_r, jacobian, joint_pos = self._get_ee_pose()  

        # Obtains the joint positions to reset. Concatenates:
        #   - the joint coordinates for the action computed by the IKDifferentialController and
        #   - the joint coordinates for the hand.
        joint_pos = torch.cat((self.controller.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos), 
                               self.default_joint_pos[:, (6):]), 
                               dim=-1)[env_ids] 
                
        # Obtains the joint velocities
        joint_vel = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.default_joint_vel[env_ids]
       
        # Writes the state to the simulation
        self.scene.articulations[self.cfg.keys[self.cfg.robot]].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


    # Resets the targets poses
    def reset_target(self, env_ids, targets, ranges):
            '''
            In:
                - env_ids - torch.tensor(m): IDs for the 'm' environments that need to be resetted.
                - targets - torch.tensor(N, ...): the target pose of the object.
                - ranges - torch.tensor(N, 2, 6): the top and bottom ranges accepatble for the new target.
            
            Out:
                - target_pose_r - torch.tensor(N, 7): new target pose as Translation+Quaternion.
                - target_pose_r_group - torch.tensor(N, ...): new target pose in the group.
                - target_pose_r_lie - torch.tensor(N,6): new target pose in the Lie Algebra.
            '''

            # Clone to get the shape
            target_pose_r = targets[0].clone()
            target_pose_r_group = targets[1].clone()
            target_pose_r_lie = targets[2].clone()

            # Randomize
            target_init_pose = sample_uniform(
                ranges[0, :, 0],
                ranges[0, :, 1],
                [self.num_envs, ranges[0, :, 0].shape[0]],
                self.device,
            )

            # Transforms Euler to quaternion
            quat = quat_from_euler_xyz(roll = target_init_pose[:, 3],
                                        pitch = target_init_pose[:, 4],
                                        yaw = target_init_pose[:, 5])
                        
            # Builds the new initial pose for the target
            target_pose_r[env_ids] = torch.cat((target_init_pose[:, :3], quat), dim = -1)[env_ids].float()
            target_pose_r_group[env_ids] = self.convert_to_group(target_init_pose[:, :3], quat)[env_ids]
            target_pose_r_lie[env_ids] = self.log(target_pose_r_group)[env_ids]

            return target_pose_r, target_pose_r_group, target_pose_r_lie


    # Resets the simulation --> Overrides method of DirectRLEnv
    def _reset_idx(self, env_ids: Sequence[int] | None):
        '''
        In: 
            - env_ids - Sequence(m): 'm' indexes of the environments that need to be resetted.

        Out:
            - None
        '''

        # Reset method from DirectRLEnv
        super()._reset_idx(env_ids)

        # Reset the count
        self.count = 0

        # --- Reset the robot ---
        # Reset the robot first to the default joint position so the IK is easier to compute afterwards
        self.reset_robot(env_ids = env_ids)

        # Reset the robot to a random Euclidean position
        self.reset_robot_ee(env_ids = env_ids)
        

        self.pose_group_r[env_ids] = self.convert_to_group(self.reset_robot_poses_r[:, :3], self.reset_robot_poses_r[:, 3:])[env_ids]
        self.robot_rot_ee_pose_r_lie[env_ids] = self.log(self.pose_group_r)[env_ids]

        # --- Reset controller ---
        self.controller.reset()
        
        # --- Reset target ---
        # Samples the new initial pose for the object
        self.target_pose_r, self.target_pose_r_group, self.target_pose_r_lie = self.reset_target(env_ids = env_ids,
                                                                                                 targets=(self.target_pose_r, 
                                                                                                          self.target_pose_r_group, 
                                                                                                          self.target_pose_r_lie),
                                                                                                 ranges = self.target_pose_ranges)

        # Samples the initial pose for the end target
        self.end_target_pose_r, self.end_target_pose_r_group, self.end_target_pose_r_lie = self.reset_target(env_ids = env_ids,
                                                                                                             targets=(self.end_target_pose_r, 
                                                                                                                      self.end_target_pose_r_group, 
                                                                                                                      self.end_target_pose_r_lie),
                                                                                                             ranges = self.target_pose_ranges2)
        

        grasp_point_obj_pos_r_rot, grasp_point_obj_quat_r_rot = combine_frame_transforms(t01 = self.target_pose_r[:, :3], q01 = self.target_pose_r[:, 3:],
                                                                                         t12 = torch.zeros_like(self.target_pose_r[:, :3]).to(self.device), q12 = self.cfg.rot_45_z_pos_quat)
        

        target_pose_r = torch.cat((grasp_point_obj_pos_r_rot, grasp_point_obj_quat_r_rot), dim = -1)
        target_pose_r_group = self.convert_to_group(grasp_point_obj_pos_r_rot, grasp_point_obj_quat_r_rot)
        target_pose_r_lie = self.log(target_pose_r_group)

        obs_rel = self.diff_operator(target_pose_r_group, self.pose_group_r)

        self.robot_rot_ee_pose_r_lie_rel = self.log(obs_rel)

        self.target_pose_r[env_ids] = target_pose_r[env_ids]
        self.target_pose_r_group[env_ids] = target_pose_r_group[env_ids]
        self.target_pose_r_lie[env_ids] = target_pose_r_lie[env_ids]


        # --- Reset previous values ---
        # Reset previous distances
        self.prev_dist[env_ids] = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)[env_ids]
        self.target_reached[env_ids] = torch.zeros(self.num_envs).bool().to(self.device)[env_ids]
        self.home_reached[env_ids] = torch.zeros(self.num_envs).bool().to(self.device)[env_ids]

        # Reset contacts
        self.contacts[env_ids] = torch.empty(self.num_envs, self.num_contacts).fill_(False).to(self.device)[env_ids]
        self.contacts_weight[env_ids] = torch.zeros(self.num_envs, self.num_contacts).to(self.device)[env_ids]
        self.contacts_w[env_ids] = torch.empty(self.num_envs).fill_(False).to(self.device)[env_ids]
        self.is_contact[env_ids] = torch.empty(self.num_envs).fill_(False).bool().to(self.device)[env_ids]
        
        
        """
            Creo que puedo quitar esto porque ya las actualizo en el "self.update_new_poses()"
        """
        # obs_rel = self.diff_operator(self.target_pose_r_group, self.pose_group_r)
        # self.robot_rot_ee_pose_r_lie_rel[env_ids] = self.log(obs_rel)[env_ids]
        

        # Updates the poses 
        # box_pos_x = (self.cfg.box_range_x[1] - self.cfg.box_range_x[0]) * torch.rand((self.num_envs)).to(self.device) + self.cfg.box_range_x[0]
        # box_pos_y = (self.cfg.box_range_y[1] - self.cfg.box_range_y[0]) * torch.rand((self.num_envs)).to(self.device) + self.cfg.box_range_y[0]


        # new_incs = self.cfg.object_increments.clone()
        
        # new_incs[:, 1] *= box_pos_x.int()
        # new_incs[:, 2] *= box_pos_y.int()



        # new_obj_pose_r = combine_frame_transforms(t01 = self.cfg.object_base_pose[:, :3], q01 = self.cfg.object_base_pose[:, 3:],
        #                                         t12 = new_incs[:, :3], q12 = new_incs[:, 3:])
        
        
        # new_obj_pose_w = combine_frame_transforms(t01 = self.root_robot_pose[:, :3], q01 = self.root_robot_pose[:, 3:],
        #                                           t12 = new_obj_pose_r[0],           q12 = new_obj_pose_r[1])
        
        
        
        # self.scene.rigid_objects["object"].write_root_pose_to_sim(root_pose = torch.cat((new_obj_pose_w), dim = -1), env_ids = env_ids)
        # self.scene.rigid_objects["object"].write_root_velocity_to_sim(root_velocity = torch.zeros(self.num_envs, 6).to(self.device), env_ids = env_ids)



        # Writes the new object position to the simulation
        self.scene.rigid_objects["object"].write_root_pose_to_sim(root_pose = torch.cat((self.target_pose_r[:, :3], 
                                                                                         self.target_pose_r[:, 3:]), dim = -1)[env_ids], env_ids = env_ids)
        self.scene.rigid_objects["object"].write_root_velocity_to_sim(root_velocity = torch.zeros((self.num_envs, 6), device=self.device)[env_ids], env_ids = env_ids)



        self.update_new_poses() 


        
        self.trajectory = []
        self.trajectory_save = []

        references = [self.interm_pose_r_lie.clone(), self.target_pose_r_lie.clone(), self.end_target_pose_r_lie.clone()]

        # NMPC model creation
        self.u_opt = torch.tensor([[0, 0, 0, 0, 0, 0]]).to(self.device)
        self.x0 = torch.tensor([[0, 0, 0, 0, 0, 0]]).to(self.device)

        self.x0 = self.gripper_pose_r_lie.clone()
        x0 = torch.cat((self.x0, self.u_opt), dim = -1)
        x0 = x0[0].cpu().numpy().tolist()

        save_idx = 0
        self.start_grip_idx = 0

        for idx, ref in enumerate(references):

            
                
            self.model, self.nmpc, ellipsoid_r_torch = drop_MPC_setup(self.cfg.obst_list, 
                                                    self.cfg.ellipsoid_r, 
                                                    ini = x0, 
                                                    ref = ref[0],
                                                    dt = self.cfg.dt, 
                                                    lie = self.cfg.lie_mpc,)

            # ======================================================
            # Simulation loop
            # ======================================================
            
            sol = self.model.solution


            # x0 = x0[0].cpu().numpy().tolist()

            for k in range(self.cfg.n_steps_mpc):
                u_opt = self.nmpc.optimize(x0)
                self.model.simulate(u=u_opt, steps=1)
                x0 = sol['x:f']

                x0_tensor = torch.tensor([[float(x0[0]), float(x0[1]), float(x0[2]), 
                                        float(x0[3]), float(x0[4]), float(x0[5])]])
                
                x0_group = self.exp(x0_tensor)
                x0_lab   = self.convert_to_Lab(x0_group)

                self.trajectory_save.append(x0_tensor[0].numpy().tolist())

                if self.cfg.get_img_mpc:
                    fig, ax = get_frame(
                        x0_tensor[0],
                        tgt=ref[0].cpu().numpy().tolist(),
                        traj=self.trajectory_save,
                        ax=None,
                        obst_centers=self.cfg.obst_list,
                        obst_radii=ellipsoid_r_torch*2,
                        rot = self.cfg.get_rot,)
            
                    name = f"{save_idx:03d}.png"
                    # fig.savefig(os.path.join(path, name))

                    name = f"{save_idx:03d}.png"
                    ax.view_init(elev=0, azim=0)
                    fig.savefig(os.path.join(self.cfg.path_traj_mpc, name))


                    f"side_{save_idx:03d}.png"
                    # ax.view_init(elev=0, azim=45)
                    # fig.savefig(os.path.join(path, name))


                    name = f"other_{save_idx:03d}.png"
                    ax.view_init(elev=90, azim=-0)
                    fig.savefig(os.path.join(self.cfg.path_traj_mpc, name))
                    
                    plt.close(fig)


                save_idx += 1
                
                if idx == 1:
                    self.start_grip_idx = save_idx

                if torch.norm(x0_tensor.to(self.device) - ref).item() < self.cfg.plan_chg_thres:
                    break

            if idx == 1:
                self.subs_limit = save_idx
            



        self.trajectory_save = torch.tensor(self.trajectory_save).to(self.device)
        self.prev_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).to(self.device)

        self.increment_condition = False
        self.gripper_action = False


        print("--- Episode: ", self.episode_id)
        self.episode_id += 1

        # saving_dir = os.path.join(self.cfg.path_traj_mpc, "traj.pkl")
        # save_traj(self.trajectory_save, lie = True, saving_dir = saving_dir)
        if not self.cfg.test:
            self.writer = HDF5EpisodeWriter(
                                            output_dir=os.path.join(self.current_path, "dataset"),
                                            episode_idx=self.episode_id,
                                            max_steps=int(self.trajectory_save.shape[0] / self.cfg.save_interval)
                                            )
        else:
            # Create model
            self.test_model = CnnPolicy(
                pose_dim=6,
                action_dim=7,
                in_channels=1,
                hidden_dim=128
            )

            # Load checkpoint
            checkpoint = torch.load(self.cfg.model_path, map_location=self.device)
            
            # If you saved only state_dict
            self.test_model.load_state_dict(checkpoint, strict = False)

            # OR if checkpoint is wrapped
            # self.test_model.load_state_dict(checkpoint["self.model_state_dict"])

            # Move to GPU if available
            self.test_model.to(self.device)

            # Inference mode
            self.test_model.eval()

