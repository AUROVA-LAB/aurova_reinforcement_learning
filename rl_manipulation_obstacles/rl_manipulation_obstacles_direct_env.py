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

import numpy as np
import matplotlib.pyplot as plt

from pynput import keyboard

import sys
sys.path.append('../../../')

from hilo_mpc import NMPC, Model
import casadi as ca




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
        if subtitles:
            ax.set_title(subtitles[idx])

    # Remove extra axes if any
    for ax in axes[n_images:]:
        fig.delaxes(ax)

    # Set title
    if title:
        plt.suptitle(title)

    # Adjust layout to fit the title
    plt.tight_layout()

    # Save the figure
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    # Close the figure
    plt.close()

def drop_NMPC_setup(obst_list, ellipsoid_r, ini = [0,0,0,0,0,0,
                                                   0,0,0,0,0,0], ref = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]):
    
    # ======================================================
    # Create model
    # ======================================================
    model = Model(plot_backend='bokeh')

    # States
    x = model.set_dynamical_states(['X', 'Y',    'Z', 'X_', 'Y_', 'Z_', 
                                    'Vx', 'Vy', 'Vz', 'Wx', 'Wy', 'Wz'])
    X = x[0]
    Y = x[1]
    Z = x[2]
    X_ = x[3]
    Y_ = x[4]
    Z_ = x[5]

    Vx = x[6]
    Vy = x[7]
    Vz = x[8]
    Wx = x[9]
    Wy = x[10]
    Wz = x[11]

    # Measurements
    model.set_measurements(['yX', 'yY', 'yZ', 'yX_', 'yY_', 'yZ_',
                            'yVx', 'yVy', 'yVz', 'yWx', 'yWy', 'yWz'])
    model.set_measurement_equations(x)

    # Inputs
    u = model.set_inputs(['ax', 'ay', 'az', 'ax_', 'ay_', 'az_'])
    ax = u[0]
    ay = u[1]
    az = u[2]
    ax_ = u[3]
    ay_ = u[4]
    az_ = u[5]

    # ======================================================
    # Dynamics
    # ======================================================
    dx = ca.vertcat(
        Vx,
        Vy,
        Vz,
        Wx,
        Wy,
        Wz,
        ax,
        ay,
        az,
        ax_,
        ay_,
        az_,
    )
    model.set_dynamical_equations(dx)





    # ======================================================
    # Obstacle avoidance via algebraic constraint
    # ======================================================

    # z = model.set_algebraic_states(['c_obs' + str(idx) for idx in range(len(obst_list))])


    # rhs = []
    
    # # Obstacle parameters
    # for idx, o in enumerate(obst_list):

    #     # Algebraic state (constraint slack, optional)

    #     # Constraint equation: c_obs = ((X-Xo)/a)^2 + ((Y-Yo)/b)^2 + ((Z - Zo)/c)^2- 1 ->
    #     '''
    #     sería algo así como:
    #         rhs = (z = ((X-Xo)/a)^2 + ((Y-Yo)/b)^2 + ((Z - Zo)/c)^2- 1)

    #     pero lo tienes que poner de manera que rhs = 0 para que entre en "set_algebraic_equations" 
    #     '''
        
    #     rhs.append((X - o[0].item())**2 / ellipsoid_r[0]**2 + (Y - o[1].item())**2 / ellipsoid_r[1]**2 + (Z - o[2].item())**2 / ellipsoid_r[2]**2 - 1 - z[idx])

    # model.set_algebraic_equations(ca.vertcat(*rhs))


    # ======================================================
    # Setup model
    # ======================================================
    dt = 0.01
    model.setup(dt=dt)

    # ======================================================
    # NMPC
    # ======================================================
    nmpc = NMPC(model)

    # Target
    # X_ref = ref[0, 0].item()
    # Y_ref = ref[0, 1].item()
    # Z_ref = ref[0, 2].item()
    # X__ref = ref[0, 3].item()
    # Y__ref = ref[0, 4].item()
    # Z__ref = ref[0, 5].item()

    # Vx_ref = 0.0
    # Vy_ref = 0.0
    # Vz_ref = 0.0
    # Wx_ref = 0.0
    # Wy_ref = 0.0
    # Wz_ref = 0.0

    # -0.8948, -0.3471,  0.8949, -0.3400,  0.0687,  0.3558

    # X_ref = -0.8800
    # Y_ref = -0.3645
    # Z_ref = 0.8800
    # X__ref = -0.3400
    # Y__ref = -0.1850
    # Z__ref = 0.3616


    X_ref = -0.6800
    Y_ref = 0.1300
    Z_ref = 0.6194
    X__ref = -1.5708
    Y__ref = 0.7854
    Z__ref = 1.5708

    Vx_ref = 0.0
    Vy_ref = 0.0
    Vz_ref = 0.0
    Wx_ref = 0.0
    Wy_ref = 0.0
    Wz_ref = 0.0

    nmpc.quad_stage_cost.add_states(
        names=['X', 'Y', 'Z', 'X_', 'Y_', 'Z_', 
                'Vx', 'Vy', 'Vz', 'Wx', 'Wy', 'Wz'],

        ref=[X_ref, Y_ref, Z_ref, X__ref, Y__ref, Z__ref,
                Vx_ref, Vy_ref, Vz_ref, Wx_ref, Wy_ref, Wz_ref],
        weights=[50, 50, 50, 50, 50, 50,
                    5, 5, 5, 5, 5, 5]
    )

    nmpc.quad_stage_cost.add_inputs(
        names=['ax', 'ay', 'az', 'ax_', 'ay_', 'az_'],
        weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )

    # Horizon
    nmpc.horizon = 5

    # Box constraints
    nmpc.set_box_constraints(
        x_lb=[-10, -10, -10, -10, -10, -10, 
            -10, -10, -10, -10, -10, -10],
        x_ub=[10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10],
        u_lb=[-0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
        u_ub=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        # z_lb=[0.0]*24,      # <-- enforces obstacle avoidance
        # z_ub=[ca.inf]*24
    )

    # Initial conditions
    print("\n\n\nNMPC initial: ", ini)
    x0 = [-0.2918,  0.1293,  0.6283, -1.5634,  0.9994,  1.5536, 0,0,0,0,0,0] # ini[0].cpu().numpy().tolist()
    z0 = [1.0]# [1.0]*24   # start feasible
    u0 = [0]*6

    model.set_initial_conditions(x0=x0)
    nmpc.set_initial_guess(x_guess=x0, u_guess=u0)

    nmpc.setup(options={'print_level': 0})

    return model, nmpc

def get_frame(x, tgt, ax=None):
    """
    x     : current state (at least X,Y,Z)
    tgt   : [Xt, Yt, Zt]
    obst  : [Xo, Yo, Zo]
    traj  : list or array of past positions [[x,y,z], ...]
    ax    : matplotlib 3D axis
    """


    X, Y, Z = float(x[0,3]), float(x[0,4]), float(x[0,5])
    Xt, Yt, Zt = tgt

    # Create axis if needed
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()
        ax.cla()



    # =========================
    # Plot current robot pose
    # =========================
    ax.scatter(X, Y, Z, c='k', s=80, label="Robot", zorder=5)

    # =========================
    # Plot target
    # =========================
    ax.scatter(Xt, Yt, Zt, c='g', s=100, label="Target", zorder=5)

    # =========================
    # Plot obstacle
    # =========================

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
    ax.set_title("3D Robot Trajectory")
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

        self.object_pose_w_lab = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).repeat(self.num_envs, 1).to(self.device)
        
        self.robot_rot_ee_pose_r_lie_rel = torch.zeros((self.num_envs, self.cfg.size)).to(self.device).float()
        self.robot_rot_ee_pose_r_lie = torch.zeros((self.num_envs, self.cfg.size)).to(self.device).float()

        self.root_robot_pose = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.root_state_w[:, 0:7]
        

        # Indexes for: robot joints, hand joints, all joints
        self._robot_joints_idx = self.scene.articulations[self.cfg.keys[self.cfg.robot]].find_joints(self.cfg.joints[self.cfg.robot])[0]
        self._hand_joints_idx = self.scene.articulations[self.cfg.keys[self.cfg.robot]].find_joints(self.cfg.hand_joints[self.cfg.robot])[0]
        self._all_joints_idx = self.scene.articulations[self.cfg.keys[self.cfg.robot]].find_joints(self.cfg.all_joints[self.cfg.robot])[0]

        # IK Controller
        controller_cfg = DifferentialIKControllerCfg(command_type = "pose", use_relative_mode = False, ik_method = "dls")
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


        # Obtain the ranges in which sample reset poses
        self.ee_pose_ranges = torch.tensor([[ [(i + cfg.apply_range[idx]*inc[0]), (i + cfg.apply_range[idx]*inc[1])] for i, inc in zip(poses, cfg.ee_pose_incs)] for idx, poses in enumerate(cfg.ee_init_pose)]).to(self.device)
        self.target_pose_ranges = torch.tensor([[ [(i + cfg.apply_range_tgt*inc[0]), (i + cfg.apply_range_tgt*inc[1])] for i, inc in zip(poses, cfg.target_poses_incs)] for poses in cfg.target_pose]).to(self.device)
        
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
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()  # ✅ No bloquea
        # --------------------------------


        # --- Camera poses ---
        self.cfg.camera_ext_trans, self.cfg.camera_ext_rot = combine_frame_transforms(t01 = self.root_robot_pose[:, :3],     q01 = self.root_robot_pose[:, 3:7],
                                                                                      t12  =self.cfg.camera_ext_trans,   q12 = self.cfg.camera_ext_rot)
        
        self.cfg.camera_ext_trans_2, self.cfg.camera_ext_rot_2 = combine_frame_transforms(t01 = self.root_robot_pose[:, :3],     q01 = self.root_robot_pose[:, 3:7],
                                                                                      t12  =self.cfg.camera_ext_trans_2,   q12 = self.cfg.camera_ext_rot_2)
        

        new_ext_pos, new_ext_rot = combine_frame_transforms(t01 =self.cfg.camera_ext_trans,   q01 = self.cfg.camera_ext_rot,
                                                            t12 = torch.zeros_like(self.cfg.camera_ext_trans).to(self.device),   q12 = self.cfg.rot_neg90_xy)
        self.scene.sensors["camera_ext"].set_world_poses(positions = new_ext_pos, orientations = new_ext_rot)
   
        
        
        new_ext_pos_2, new_ext_rot_2 = combine_frame_transforms(t01 =self.cfg.camera_ext_trans_2,   q01 = self.cfg.camera_ext_rot_2,
                                                            t12 = torch.zeros_like(self.cfg.camera_ext_trans_2).to(self.device),   q12 = self.cfg.rot_neg90_xy_2)
        self.scene.sensors["camera_ext_2"].set_world_poses(positions = new_ext_pos_2, orientations = new_ext_rot_2)


        self.contact_thres = self.cfg.contact_matrix[0, :-1].mean().item()*2

        self.u_opt = torch.tensor([[0, 0, 0, 0, 0, 0]]).to(self.device)



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
        
        self.scene.rigid_objects["shelf"] = RigidObject(self.cfg.shelf_cfg)
        self.scene.rigid_objects["object"] = RigidObject(self.cfg.object_cfg)

        # Add extras (markers, ...)
        self.scene.extras["markers"] = VisualizationMarkers(self.cfg.marker_cfg)

        self.scene.sensors["camera"] = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["camera_ext"] = TiledCamera(self.cfg.tiled_camera_ext)
        self.scene.sensors["camera_ext_2"] = TiledCamera(self.cfg.tiled_camera_ext_2)

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
        actions = actions[:, :-1]

        # Perform increment in the algebra and exponential map -> (plus operator)
        action_pose = self.exp(self.robot_rot_ee_pose_r_lie_rel + actions)
        action_pose = self.mul_operator(self.target_pose_r_group, action_pose)
        action_pose = self.normalize(action_pose)

        # Convert to IsaacLab representation (translation, quaternion)
        action_pose_lab = self.convert_to_Lab(action_pose)



        grip_action = self.prim_g_action
        action_pose_lab[:, :3] += self.prim_action[:, :3]
        self.prim_action = torch.zeros_like(self.prim_action).to(self.device)
        self.prim_g_action = torch.zeros_like(self.prim_g_action).to(self.device)


        # print(self.robot_rot_ee_pose_r_lie[0])
        # print(self.robot_rot_ee_pose_r_lie[0].cpu().numpy())


        robot_pose = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.body_state_w[:, self.ee_jacobi_idx+1, 0:7]
        obj_pose = self.debug_target_pose_w

        
        object_euler = torch.tensor(euler_xyz_from_quat(obj_pose[:, 3:])).unsqueeze(0).to(self.device)
        robot_euler = torch.tensor(euler_xyz_from_quat(robot_pose[:, 3:])).unsqueeze(0).to(self.device)
        

        vel_pos = torch.cat((robot_pose[:, :3], robot_euler, self.u_opt), dim = -1)

        print("Object Pose: ", torch.cat((obj_pose[:, :3], object_euler, self.u_opt), dim = -1))

        u_opt = self.nmpc.optimize(vel_pos[0].cpu().numpy())
        self.model.simulate(u=u_opt, steps=1)
        x0 = self.model.solution['x:f']

        # print(torch.tensor([[float(x0[3]), float(x0[4]), float(x0[5]), 
        #                      float(x0[0]), float(x0[1]), float(x0[2])]]).to(self.device))

        x0 = torch.tensor([[float(x0[0]), float(x0[1]), float(x0[2]), 
                            float(x0[3]), float(x0[4]), float(x0[5])]]).to(self.device)
        self.u_opt = torch.tensor([[float(u_opt[0]), float(u_opt[1]), float(u_opt[2]), 
                                    float(u_opt[3]), float(u_opt[4]), float(u_opt[5])]]).to(self.device)
        
        print("Robot Pose: ", vel_pos)
        print("Command: ", x0)
        print("u_opt: ", self.u_opt)
        print("-----")
        
        
        
        quat = quat_from_euler_xyz(roll = x0[:, 3], pitch = x0[:, 4], yaw = x0[:, 5])

        x0 = torch.cat((x0[:, :3], quat), dim = -1)



        fig, ax = get_frame(
            x0,
            tgt=[-0.3400, 0.0687, 0.3558],
            ax=None
        )

        self.count += 1

        fig.savefig(f"{self.count:03d}.png")
        plt.close(fig)

        





        # Set the command for the IKDifferentialController
        self.controller.set_command(x0)
                
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
        actions = self._preprocess_actions(actions)

        # Obtains the increments and the poses
        self.perform_increment(actions = actions)


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
        
        # print(self.scene.rigid_objects["shelf"].data.body_state_w[:, 0, :3])
        shelf_pose = self.scene.rigid_objects["shelf"].data.body_state_w[:, 0, :7]
        # shelf_pose[:, 0] -= 0.15
        # shelf_pose[:, 1] -= 0.15
        # shelf_pose[:, 2] += 0.25
        # print("Posicion balda:", shelf_pose[:, :3])

        '''
        Pose de la estanteria:
            · -6.0000e-01, -6.0000e-01, -1.1921e-07
        
        Pose de eslabones:
            · X: 0.1        -> ancho            -> semi-eje  = 0.1
            · Y: -0.15      -> profundidad      -> semi-eje  = 0.15
            · Z: +0.25       -> alto             -> semi-eje = 0.25
            · Rango de seguridad -> 0.1 + diámetro de la esfera alrededor de la pinza
        '''
        

        # Obtains a tensor of indices (a tensor containing tensors from 0 to the number of markers)
        marker_indices = torch.arange(self.scene.extras["markers"].num_prototypes).repeat(self.num_envs)

        # Updates poses in simulation
        self.scene.extras["markers"].visualize(translations = torch.cat((self.debug_robot_ee_pose_w[:, :3], 
                                                                         self.debug_target_pose_w[:, :3],
                                                                         shelf_pose[:, :3],
                                                                         self.cfg.shelf_poses[:, :3])), 
                                                                         
                                                orientations = torch.cat((self.debug_robot_ee_pose_w[:, 3:], 
                                                                          self.debug_target_pose_w[:,3:],
                                                                          shelf_pose[:, 3:],
                                                                          self.cfg.shelf_poses[:, 3:])), 

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
        robot_rot_ee_pos_r, robot_rot_ee_quat_r = subtract_frame_transforms(t01 = self.root_robot_pose[:, :3], q01 = self.root_robot_pose[:, 3:],
                                                                              t02 = self.debug_robot_ee_pose_w[:, :3], q02 = self.debug_robot_ee_pose_w[:, 3:])


        # Fix double cover
        neg_idx = robot_rot_ee_quat_r[:, 0] < 0.0
        robot_rot_ee_quat_r[neg_idx] *= -1
        


        # --- Target pose ---
        # Get object world pose
        self.debug_target_pose_w = self.scene.rigid_objects["object"].data.body_state_w[:, 0, 0:7]

        # Obtain the relative pose w.r.t. the robot root frame
        target_pos_r, target_quat_r = subtract_frame_transforms(t01 = self.root_robot_pose[:, :3], q01 = self.root_robot_pose[:, 3:],
                                                                t02 = self.debug_target_pose_w[:, :3], q02 = self.debug_target_pose_w[:, 3:])
        
        target_pos_r, target_quat_r = combine_frame_transforms(t01 = target_pos_r,                   q01 = target_quat_r,
                                                               t12 = self.cfg.object_translation,    q12 = self.cfg.object_rotation)
        
        target_pos_w, target_quat_w = combine_frame_transforms(t01 = self.root_robot_pose[:, :3],        q01 = self.root_robot_pose[:, 3:],
                                                               t12 = target_pos_r,                       q12 = target_quat_r)
        
        self.debug_target_pose_w = torch.cat((target_pos_w, target_quat_w), dim = -1)


        # Fix double cover
        neg_idx = target_quat_r[:, 0] < 0.0
        target_quat_r[neg_idx] *= -1



        # --- Build relative pose observation ---
        # Build the group object
        self.pose_group_r = self.convert_to_group(robot_rot_ee_pos_r, robot_rot_ee_quat_r)
        self.target_pose_r_group = self.convert_to_group(target_pos_r, target_quat_r)

        # print(self.log(self.target_pose_r_group))

        # Transform to the Lie algebra
        self.robot_rot_ee_pose_r_lie = self.log(self.pose_group_r)
        self.target_pose_r_lie = self.log(self.target_pose_r_group)
        diff = self.diff_operator(self.target_pose_r_group, self.pose_group_r)
        self.robot_rot_ee_pose_r_lie_rel = self.log(diff)


        

        # Hand observations
        self.hand_joints_pos = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.joint_pos[:, self._hand_joints_idx]
        self.hand_pose = torch.round(self.hand_joints_pos[:, 2] / self.cfg.m[0], decimals = 0) / 140.0



        # Put actual end effector pose on the gripper
        robot_rot_ee_pos_r, robot_rot_ee_quat_r = combine_frame_transforms(t01 = robot_rot_ee_pos_r,       q01 = robot_rot_ee_quat_r,
                                                                           t12 = self.cfg.ee_translation,  q12 = self.cfg.ee_rotation)
        
        self.pose_group_r = self.convert_to_group(robot_rot_ee_pos_r, robot_rot_ee_quat_r)

        robot_rot_ee_pos_w, robot_rot_ee_quat_w = combine_frame_transforms(t01 = self.root_robot_pose[:, :3], q01 = self.root_robot_pose[:, 3:],
                                                                           t12 = robot_rot_ee_pos_r,          q12 = robot_rot_ee_quat_r)

        self.debug_robot_ee_pose_w = torch.cat((robot_rot_ee_pos_w, robot_rot_ee_quat_w), dim = -1)



    def _get_images(self, camera_key = "camera"):
        '''
        In:
            - camera_key - str: key of the camera to obtain images from.
        
        Out:
            - imgs - torch.Tensor: image obtained from the desired camera.
        '''

        self.count += 1
        output_dir = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/aurova_reinforcement_learning/"

        
        camera_pose = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.body_state_w[:, self.camera_idx, 0:7]
        
        new_camera_trans, new_camera_rot = combine_frame_transforms(t01 = camera_pose[:, :3],     q01 = camera_pose[:, 3:7],
                                                                    t12 = self.cfg.camera_trans,  q12 = self.cfg.camera_rot)
        
        self.scene.sensors[camera_key].set_world_poses(positions = new_camera_trans, orientations = new_camera_rot)

        
        
   
        # Obtain images from the sensor
        image_tensor = self.scene.sensors[camera_key].data.output["depth"][:, ..., :3].permute(0, 3, 1, 2).reshape(self.num_envs, -1)
        image_tensor_ext = self.scene.sensors["camera_ext"].data.output["depth"][:, ..., :3].permute(0, 3, 1, 2).reshape(self.num_envs, -1)
        image_tensor_ext_2 = self.scene.sensors["camera_ext_2"].data.output["depth"][:, ..., :3].permute(0, 3, 1, 2).reshape(self.num_envs, -1)

        image_tensor = torch.cat((image_tensor, image_tensor_ext, image_tensor_ext_2), dim = -1)
        image_tensor_ = self.scene.sensors[camera_key].data.output["instance_id_segmentation_fast"][:, ..., :3]


        # Render images every certain amount of steps
        if self.cfg.save_imgs:
            if self.count % 5 == 0:
            # Function to save images (in utils)
                save_images_grid(images = [image_tensor_[0]],
                                 subtitles = ["Camera"],
                                 title = "RGB Image: Cam0",
                                 filename = os.path.join(output_dir, "rgb", f"{self.count:04d}.jpg"))
                
        return image_tensor




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
        
        image = self._get_images()
        # image[image == float('inf')] = 255.0
        # image /= 255.0

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

        # --- Contacts ---
        contacts_weight = (self.contacts * self.cfg.contact_matrix)
        contacts_w = contacts_weight.sum(-1)

        is_contact = (contacts_weight[:, :-1] > self.contact_thres).sum(-1)


        aux_tgt_reached = self.target_reached.clone()

        # Target reached flag
        self.target_reached = torch.logical_or(torch.logical_and(dist_target < self.cfg.distance_thres, is_contact), self.target_reached)
        self.home_reached = dist_home < self.cfg.distance_thres

        apply_bonus = torch.logical_and(self.target_reached, torch.logical_not(aux_tgt_reached))


        # ---- Distance reward ----
        # Reward for the approaching
        reward = mod * self.cfg.rew_scale_dist * torch.exp(-2*dist) + contacts_w

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
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Checks out of bounds in velocity
        out_of_bounds = torch.norm(self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.body_state_w[:, self.ee_jacobi_idx+1, 7:], dim = -1) > self.cfg.velocity_limit 

        # Truncated and terminated variables
        truncated = out_of_bounds
        terminated = torch.logical_or(time_out*0, self.home_reached)

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
        # Sample a random pose for the target
        target_init_pose = sample_uniform(
            self.target_pose_ranges[0, :, 0],
            self.target_pose_ranges[0, :, 1],
            [self.num_envs, self.target_pose_ranges[0, :, 0].shape[0]],
            self.device,
        )

        # Transforms Euler to quaternion
        quat = quat_from_euler_xyz(roll = target_init_pose[:, 3],
                                    pitch = target_init_pose[:, 4],
                                    yaw = target_init_pose[:, 5])
        
        neg_idx = quat[:, 0] < 0.0
        quat[neg_idx] *= -1

        # Builds the new initial pose for the target
        self.target_pose_r[env_ids] = torch.cat((target_init_pose[:, :3], quat), dim = -1)[env_ids].float()

        self.target_pose_r_group[env_ids] = self.convert_to_group(target_init_pose[:, :3], quat)[env_ids]
        self.target_pose_r_lie[env_ids] = self.log(self.target_pose_r_group)[env_ids]

        # --- Reset previous values ---
        # Reset previous distances
        self.prev_dist[env_ids] = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)[env_ids]
        self.target_reached[env_ids] = torch.zeros(self.num_envs).bool().to(self.device)[env_ids]
        self.home_reached[env_ids] = torch.zeros(self.num_envs).bool().to(self.device)[env_ids]

        # Reset contacts
        self.contacts[env_ids] = torch.empty(self.num_envs, self.num_contacts).fill_(False).to(self.device)[env_ids]
        
        obs_rel = self.diff_operator(self.target_pose_r_group, self.pose_group_r)

        self.robot_rot_ee_pose_r_lie_rel[env_ids] = self.log(obs_rel)[env_ids]
        
        # Updates the poses 
        self.update_new_poses() 


        box_pos_x = (self.cfg.box_range_x[1] - self.cfg.box_range_x[0]) * torch.rand((self.num_envs)).to(self.device) + self.cfg.box_range_x[0]
        box_pos_y = (self.cfg.box_range_y[1] - self.cfg.box_range_y[0]) * torch.rand((self.num_envs)).to(self.device) + self.cfg.box_range_y[0]


        new_incs = self.cfg.object_increments.clone()
        
        new_incs[:, 1] *= box_pos_x.int()
        new_incs[:, 2] *= box_pos_y.int()



        new_obj_pose_r = combine_frame_transforms(t01 = self.cfg.object_base_pose[:, :3], q01 = self.cfg.object_base_pose[:, 3:],
                                                t12 = new_incs[:, :3], q12 = new_incs[:, 3:])
        
        
        new_obj_pose_w = combine_frame_transforms(t01 = self.root_robot_pose[:, :3], q01 = self.root_robot_pose[:, 3:],
                                                  t12 = new_obj_pose_r[0], q12 = new_obj_pose_r[1])
        
        
        
        self.scene.rigid_objects["object"].write_root_pose_to_sim(root_pose = torch.cat((new_obj_pose_w), dim = -1), env_ids = env_ids)
        self.scene.rigid_objects["object"].write_root_velocity_to_sim(root_velocity = torch.zeros(self.num_envs, 6).to(self.device), env_ids = env_ids)

        
        obj_pose_r = self.log(self.convert_to_group(new_obj_pose_w[0], new_obj_pose_w[1]))
        
        vel = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.body_state_w[:, self.ee_jacobi_idx+1, 7:]
        vel_pos = torch.cat((self.robot_rot_ee_pose_r_lie, vel), dim = -1)

        self.u_opt = torch.tensor([[0, 0, 0, 0, 0, 0]]).to(self.device)

        # NMPC model creation
        self.model, self.nmpc = drop_NMPC_setup(self.cfg.shelf_poses[:, :3], self.cfg.ellipsoid_r, ini = vel_pos, ref = obj_pose_r)

        # print(self.cfg.shelf_poses[:, :3])
        # print(self.cfg.ellipsoid_r)
        # print(vel_pos)
        # print(obj_pose_r)
        # raise

        



