from __future__ import annotations

import os
import torch
from collections.abc import Sequence
import copy

from .mdp.utils import compute_rewards, save_images_grid, update_seq
from .mdp.rewards import dual_quaternion_error 
from .rl_manipulation_direct_env_cfg import RLManipulationDirectCfg, update_cfg

from .py_dq.src.dq import *
from .py_dq.src.distances import *
from .py_dq.src.lie import *
from .py_dq.src.interpolators import *

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.sensors import ContactSensor, Camera
from isaaclab.markers import VisualizationMarkers
from isaaclab.assets import RigidObject

'''
                    ############## IMPORTANT #################
   The whole environment is build for two robots: the UR5e and Kinova GEN3-7dof.
   These two variables (cfg.UR5e and cfg.GEN3) serve as an abstraction to treat the robots during the episodes. In fact,
all the methods need an index to differentiate from which robot get the information.
   Also, data storage is performed using lists, not tensors because the joint space of the robots is
different from one another.
'''

# Class for the Bimanual Direct Environment
class RLManipulationDirect(DirectRLEnv):
    cfg: RLManipulationDirectCfg

    # --- init function ---
    def __init__(self, cfg: RLManipulationDirectCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # --- Debug variables ---
        # Debug poses for the object and end effector of the GEN3 robot. These poses 
        # are used to draw the markers in the simulation
        self.debug_robot_ee_pose_w = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)
        self.debug_target_pose_w = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)

        # Poses for the object and GEN3 robot so they can match when performing the grasping
        # self.robot_rot_ee_pose_r = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)
        # self.robot_rot_ee_pose_r_lie = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)
        self.target_pose_r =  torch.tensor([0.0 ,0.0 ,0.0, 1.0 ,0.0 ,0.0 ,0.0]).to(self.device).repeat(self.num_envs, 1).float()
        self.target_pose_r_lie =  torch.zeros((self.num_envs, cfg.size)).to(self.device).float()
        
        self.obs_seq_robot_pose_r = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, self.cfg.seq_len, 1).to(self.device).float()
        self.obs_seq_robot_pose_r_lie = torch.zeros((self.num_envs, self.cfg.seq_len, self.cfg.size)).to(self.device).float()

        # Indexes for: robot joints, hand joints, all joints
        self._robot_joints_idx = self.scene.articulations[self.cfg.keys[self.cfg.robot]].find_joints(self.cfg.joints[self.cfg.robot])[0]
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
       

        # List for the default joint poses of both robots --> As a list due to the different joints of the arms (6 and 7) 
        self.default_joint_pos = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.default_joint_pos


        # List of joint actions
        self.actions = copy.deepcopy(self.default_joint_pos)

        # Poses obtained at reset
        self.reset_robot_poses_r = torch.zeros((self.num_envs, 7)).to(self.device) 

        # Update configuration class
        self.cfg = update_cfg(cfg = cfg, num_envs = self.num_envs, device = self.device)

        # Obtain the ranges in which sample reset poses
        self.ee_pose_ranges = torch.tensor([[ [(i + cfg.apply_range[idx]*inc[0]), (i + cfg.apply_range[idx]*inc[1])] for i, inc in zip(poses, cfg.ee_pose_incs)] for idx, poses in enumerate(cfg.ee_init_pose)]).to(self.device)
        self.target_pose_ranges = torch.tensor([[ [(i + inc[0]), (i + inc[1])] for i, inc in zip(poses, cfg.target_poses_incs)] for poses in cfg.target_pose]).to(self.device)


        # Create output directory to save images
        if self.cfg.save_imgs:
            self.output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
            os.makedirs(self.output_dir, exist_ok=True)

        # Previous distance
        self.prev_dist = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)

        # Target reached flag
        self.target_reached = torch.zeros(self.num_envs).to(self.device).bool()

        # --- Lie algebra ---
        # List of mappings
        map_list = [[[identity_map, identity_map], [exp_bruno, log_bruno], [exp_stereo, log_stereo]],
                [[identity_map, identity_map]],
                [[identity_map, identity_map]],
                [[identity_map, identity_map]],
                [[identity_map, identity_map]]]

        # List of conversions
        conversions = [convert_dq, None, None, None, None]

        # List of interpolators
        interpolators = [ScLERP, None, None, None, None]

        # Lis of distance functions
        distances = [[dqLOAM_distance, geodesic_dist, double_geodesic_dist],
                     [None],
                     [None],
                     [None],
                     [None]]

        # Assign the functions according to configuration
        self.exp = map_list[cfg.representation][cfg.mapping][0]                 # Exponential mapping
        self.log = map_list[cfg.representation][cfg.mapping][1]                 # Logarithmic mapping
        self.convert = conversions[cfg.representation]                          # Conversion Lie group to IsaacLab representation
        self.interpolator = interpolators[cfg.representation]                   # Interpolator function
        self.dist_function = distances[cfg.representation][cfg.distance]        # Distance function
    
    
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

        # Add extras (markers, ...)
        self.scene.extras["markers"] = VisualizationMarkers(self.cfg.marker_cfg)

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    # Method to preprocess the actions so they have a proper format
    def _preprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        '''
        In:
            - actions - torch.Tensor (N, m): incremental actions in the Lie algebra

        Out:
            - actions - torch.Tensor: preprocessed actions.
        '''

        # Clamp actions
        actions = torch.clamp(actions, -self.cfg.action_scaling, self.cfg.action_scaling)

        return actions
    
    
    # Obtain the end effector pose of the robot in the base frame
    def _get_ee_pose(self):
        '''
        In: 
            - idx - int(0,1): index of the robot.

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



        # Perform increment in the algebra and exponential map
        action_pose = self.exp(self.obs_seq_robot_pose_r_lie[:, -1] + actions)

        # Convert to IsaacLab representation (translation, quaternion)
        action_pose_lab = self.convert(action_pose)

        # Set the command for the IKDifferentialController
        self.controller.set_command(action_pose_lab)
                
        # Obtains the poses
        ee_pos_r, ee_quat_r, jacobian, joint_pos = self._get_ee_pose()
        
        # Get the actions for the robot. Concatenates:
        #   - the joint coordinates for the action computed by the IKDifferentialController and
        #   - the joint coordinates for the hand.
        self.actions = self.controller.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos)
        
    
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
        
        # Obtains the positions of the of the robots
        ee_pose_w_1 = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.body_state_w[:, self.ee_jacobi_idx+1, 0:7]

        # Obtains a tensor of indices (a tensor containing tensors from 0 to the number of markers)
        marker_indices = torch.arange(self.scene.extras["markers"].num_prototypes).repeat(self.num_envs)

        # Updates poses in simulation
        self.scene.extras["markers"].visualize(translations = torch.cat((ee_pose_w_1[:, :3], 
                                                                         
                                                                         self.debug_target_pose_w[:, :3],)), 
                                                orientations = torch.cat((ee_pose_w_1[:, 3:], 
                                                                          
                                                                          self.debug_target_pose_w[:,3:],),), 
                                                marker_indices=marker_indices)


    # Updates the poses of the object and robot so they can match when performing the grasp
    def update_new_poses(self):
        '''
        In:
            - None
        
        Out:
            - None
        '''

        # --- Robot poses ---
        # Obtain the pose of the GEN3 end effector in world frame
        self.debug_robot_ee_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.body_state_w[:, self.ee_jacobi_idx+1, 0:7]

        # Obtains the pose of the base of the GEN3 robot in the world frame
        robot_root_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.root_state_w[:, 0:7]

        # Obtain the pose of the end effector in GEN3 root frame
        robot_rot_ee_pos_r, robot_rot_ee_quat_r = subtract_frame_transforms(t01 = robot_root_pose_w[:, :3], q01 = robot_root_pose_w[:, 3:],
                                                                              t02 = self.debug_robot_ee_pose_w[:, :3], q02 = self.debug_robot_ee_pose_w[:, 3:])
        robot_rot_ee_pose_r = torch.cat((robot_rot_ee_pos_r, robot_rot_ee_quat_r), dim = -1)
        
        # Build the group object
        dq_pose_r = dq_from_tr(t = robot_rot_ee_pos_r, r = robot_rot_ee_quat_r)

        # Transform to the Lie algebra
        robot_rot_ee_pose_r_lie = self.log(dq_pose_r)

        # Update sequences
        self.obs_seq_robot_pose_r = update_seq(new_obs = robot_rot_ee_pose_r, seq = self.obs_seq_robot_pose_r)
        self.obs_seq_robot_pose_r_lie = update_seq(new_obs = robot_rot_ee_pose_r_lie, seq = self.obs_seq_robot_pose_r_lie)


        # --- Target pose ---
        # Obtain the pose for the target in the world frame
        grasp_point_obj_pos_r, grasp_point_obj_quat_r = combine_frame_transforms(t01 = robot_root_pose_w[:, :3], q01 = robot_root_pose_w[:, 3:],
                                                                                  t12 = self.target_pose_r[:, :3], q12 = self.target_pose_r[:, 3:])
        self.debug_target_pose_w = torch.cat((grasp_point_obj_pos_r, grasp_point_obj_quat_r), dim = -1)


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
        
        # Builds the tensor with all the observations in a single row tensor (N, 7+7+1)
        obs = torch.cat(
            (
                self.obs_seq_robot_pose_r_lie, 
                self.target_pose_r_lie.unsqueeze(1), 
            ),
            dim = 1
        )


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
        traj = self.interpolator(self.obs_seq_robot_pose_r[:, -1], self.target_pose_r, 0.1)

        dist = self.dist_function(self.obs_seq_robot_pose_r[:, -1], traj[:, 1], self.device)

        # Obtains wether the agent is approaching or not
        mod = 2*(dist < self.prev_dist).int() - 1

        # Target reached flag
        self.target_reached = dist < self.cfg.distance_thres


        # ---- Distance reward ----
        # Reward for the approaching
        reward = mod * self.cfg.rew_scale * torch.exp(-2*dist)
        

        # ---- Reward composition ----
        # Phase reward plus bonuses
        reward = reward + self.cfg.tgt_obj_reached * self.target_reached

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
        terminated = torch.logical_or(time_out, self.target_reached)

        return truncated, terminated
    
    
    # Resets the robot JOINT positions
    def reset_robot(self, env_ids):
        '''
        In:
            - idx - int(0 or 1): index for the robot.
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
            - idx - int(0 or 1): index for the robot.
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

        # Reset the sequences of poses
        self.obs_seq_robot_pose_r[env_ids] = torch.repeat_interleave(self.reset_robot_poses_r, 
                                                                     self.cfg.seq_len, 
                                                                     dim=0).view(self.num_envs,self.cfg.seq_len,-1)[env_ids]

        self.obs_seq_robot_pose_r_lie[env_ids] = torch.repeat_interleave(self.log(dq_from_tr(t = self.reset_robot_poses_r[:, :3], 
                                                                                             r = self.reset_robot_poses_r[:, 3:])), 
                                                                        self.cfg.seq_len, 
                                                                        dim=0).view(self.num_envs,self.cfg.seq_len,-1)[env_ids]

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
        
        robot_root_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.root_state_w[:, 0:7]

        tgt_pos_w, tgt_quat_w = combine_frame_transforms(t01 = robot_root_pose_w[:, :3], q01 = robot_root_pose_w[:, 3:],
                                                         t12 = target_init_pose[:, :3], q12 = quat)

        # Builds the new initial pose for the target
        self.target_pose_r[env_ids] = torch.cat((target_init_pose[:, :3], quat), dim = -1)[env_ids].float()
        self.target_pose_r_lie[env_ids] = self.log(dq_from_tr(t = tgt_pos_w, r = tgt_quat_w))[env_ids]

        # --- Reset previous values ---
        # Reset previous distances
        self.prev_dist[env_ids] = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)[env_ids]
        self.target_reached[env_ids] = torch.zeros(self.num_envs).bool().to(self.device)[env_ids]

        # Updates the poses 
        self.update_new_poses()
        
