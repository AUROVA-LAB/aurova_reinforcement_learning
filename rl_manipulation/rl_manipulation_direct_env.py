from __future__ import annotations

import os
import torch
from collections.abc import Sequence
import copy

from .mdp.utils import compute_rewards, save_images_grid, update_seq
from .mdp.rewards import dual_quaternion_error 
from .rl_manipulation_direct_env_cfg import RLManipulationDirectCfg, update_cfg, update_collisions

from .py_dq.src.dq import *
from .py_dq.src.distances import *
from .py_dq.src.lie import *
from .py_dq.src.interpolators import *

from .py_dq.src.quat_trans_lie import *
from .py_dq.src.matrix_lie import *
from .py_dq.src.euler import *

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, combine_frame_transforms
from omni.isaac.lab.utils.math import quat_from_euler_xyz
from omni.isaac.lab.sensors import ContactSensor, Camera
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.assets import RigidObject

from numpy import pi

from stable_baselines3 import PPO


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
        self.debug_target_pose_w2 = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)

        # Poses for the object and GEN3 robot so they can match when performing the grasping
        self.target_pose_r =  torch.tensor([0.0 ,0.0 ,0.0, 1.0 ,0.0 ,0.0 ,0.0]).to(self.device).repeat(self.num_envs, 1).float()
        self.target_pose_r_group =  torch.zeros((self.num_envs, cfg.size_group)).to(self.device).float()
        self.target_pose_r_lie = torch.zeros((self.num_envs, cfg.size)).to(self.device).float()

        self.target_pose_r_180 =  torch.tensor([0.0 ,0.0 ,0.0, 1.0 ,0.0 ,0.0 ,0.0]).to(self.device).repeat(self.num_envs, 1).float()
        self.target_pose_r_group_180 =  torch.zeros((self.num_envs, cfg.size_group)).to(self.device).float()
        self.target_pose_r_lie_180 = torch.zeros((self.num_envs, cfg.size)).to(self.device).float()


        self.target_pose_r2 =  torch.tensor([0.0 ,0.0 ,0.0, 1.0 ,0.0 ,0.0 ,0.0]).to(self.device).repeat(self.num_envs, 1).float()
        self.target_pose_r_group2 =  torch.zeros((self.num_envs, cfg.size_group)).to(self.device).float()
        self.target_pose_r_lie2 = torch.zeros((self.num_envs, cfg.size)).to(self.device).float()
        
        # self.obs_seq_robot_pose_r_lie_rel = torch.zeros((self.num_envs, self.cfg.seq_len, self.cfg.size)).to(self.device).float()
        self.robot_rot_ee_pose_r_lie_rel = torch.zeros((self.num_envs, self.cfg.size)).to(self.device).float()
        self.robot_rot_ee_pose_r_lie = torch.zeros((self.num_envs, self.cfg.size)).to(self.device).float()
        
        self.obs_seq_vel_lie = torch.zeros((self.num_envs, self.cfg.seq_len, 6)).to(self.device).float()
        self.obs_seq_pose_lie_rel = torch.zeros((self.num_envs, self.cfg.seq_len, self.cfg.size)).to(self.device).float()
        # self.robot_rot_ee_pose_r_lie_seq = torch.zeros((self.num_envs, self.cfg.seq_len, self.cfg.size)).to(self.device).float()

        self.hand_joints_pos = torch.zeros((self.num_envs, len(self.cfg.hand_joints[self.cfg.robot]))).float().to(self.device)
        self.hand_pose = torch.zeros((self.num_envs)).float().to(self.device)

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
       

        # List for the default joint poses of both robots --> As a list due to the different joints of the arms (6 and 7) 
        self.default_joint_pos = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.default_joint_pos
        self.default_vel = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.default_joint_vel


        # List of joint actions
        self.actions = copy.deepcopy(self.default_joint_pos)

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
        self.target_pose_ranges2 = torch.tensor([[ [(i + cfg.apply_range_tgt*inc[0]), (i + cfg.apply_range_tgt*inc[1])] for i, inc in zip(poses, cfg.target_poses_incs2)] for poses in cfg.target_pose]).to(self.device)
        
        self.z_displ = torch.tensor([0.0, 0.0, -0.21]).to(self.device).repeat(self.num_envs, 1)

        # Create output directory to save images
        if self.cfg.save_imgs:
            self.output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
            os.makedirs(self.output_dir, exist_ok=True)

        # Previous distance
        self.prev_dist = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)

        # Target reached flag
        self.target_reached = torch.zeros(self.num_envs).to(self.device).bool()
        self.height_reached = torch.zeros(self.num_envs).to(self.device).bool()

        # --- Lie algebra ---
        # List of mappings
        map_list = [[[identity_map, identity_map], [exp_bruno, log_bruno],       [exp_stereo, log_stereo]],
                    [[identity_map, identity_map],],
                    [[identity_map, identity_map], [exp_quat_cayley, log_quat_cayley], [exp_quat_stereo, log_quat_stereo]],
                    [[identity_map, identity_map], [exp_mat, log_mat]]]

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
        
        self.identities = [[1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
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
    
        self.pose_group_r = torch.tensor(self.identities[cfg.representation]).to(self.device).repeat(self.num_envs, 1).float()
        self.target_pose_r_group2 = torch.tensor(self.identities[cfg.representation]).to(self.device).repeat(self.num_envs, 1).float()
        self.target_pose_r_group = torch.tensor(self.identities[cfg.representation]).to(self.device).repeat(self.num_envs, 1).float()

        self.seq_idx = torch.tensor([range(0, self.cfg.seq_len - 1), range(1, self.cfg.seq_len)])

        # teacher_path = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/aurova_reinforcement_learning/rl_manipulation/train/logs/sb3/Isaac-RL-Manipulation-Direct-reach-v0"
        teacher_path = "/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/rl_manipulation/train/logs"

        
        self.teacher_model = PPO.load(os.path.join(teacher_path, self.cfg.path_to_pretrained))
        self.teacher_model.policy.eval()

        self.student_action = self.actions.clone()
        self.teacher_action = self.actions.clone()
    

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

        # Correct collision sensors 
        self.cfg = update_collisions(self.cfg, num_envs = self.num_envs)
        for idx, sensor_cfg in self.cfg.contact_sensors_dict.items():
            self.scene.sensors[idx] = ContactSensor(sensor_cfg)

        # Add object
        self.scene.rigid_objects["object"] = RigidObject(self.cfg.object_cfg)

        # Add extras (markers, ...)
        self.scene.extras["markers"] = VisualizationMarkers(self.cfg.marker_cfg)

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
            
            if gripper:
                actions[:, 3:-1] = torch.clamp(actions[:, 3:-1], -self.cfg.action_scaling[1], self.cfg.action_scaling[1])
                actions[:, -1] = torch.clamp(actions[:, -1], -self.cfg.grip_scaling, self.cfg.grip_scaling)

            else:
                actions[:, 3:] = torch.clamp(actions[:, 3:], -self.cfg.action_scaling[1], self.cfg.action_scaling[1])
                # actions[:, -1] = torch.clamp(actions[:, -1], -self.cfg.grip_scaling, self.cfg.grip_scaling)

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

        grip_action = actions[:, -1]#  + self.target_reached*8
        actions = actions[:, :-1]#  * torch.logical_not(self.target_reached)

        action_pose = self.exp(self.robot_rot_ee_pose_r_lie_rel + actions)
  
        action_pose = self.mul_operator(self.target_pose_r_group, action_pose)
        action_pose = self.normalize(action_pose)

        # Convert to IsaacLab representation (translation, quaternion)
        action_pose_lab = self.convert_to_Lab(action_pose)

        # Set the command for the IKDifferentialController
        # self.controller.set_command(action_pose_lab)
        self.controller.set_command(action_pose_lab)
                
        # Obtains the poses
        ee_pos_r, ee_quat_r, jacobian, joint_pos = self._get_ee_pose()
        
        # Get the actions for the robot. Concatenates:
        #   - the joint coordinates for the action computed by the IKDifferentialController and
        #   - the joint coordinates for the hand.
        self.actions[:, :6] = self.controller.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos)
        


        # --- Update gripper position ---
        actual_gripper_pos = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.joint_pos[:, self._hand_joints_idx]
        
        move_hand = (self.hand_pose*140) < 125.0

        self.actions[:, 6:] = move_hand.unsqueeze(-1) * grip_action.unsqueeze(-1) * self.cfg.moving_joints_gripper + actual_gripper_pos


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
        self.student_action = actions[:, :-1].clone()
        
        self.teacher_action = torch.tensor(self.teacher_model.predict(self.robot_rot_ee_pose_r_lie_rel.cpu().numpy(), deterministic = True)[0]).to(self.device)
        self.teacher_action = self._preprocess_actions(self.teacher_action, gripper = False)
        
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
                                                                         self.debug_target_pose_w[:, :3],
                                                                         self.debug_target_pose_w2[:, :3])), 
                                                orientations = torch.cat((ee_pose_w_1[:, 3:], 
                                                                          self.debug_target_pose_w[:, 3:],
                                                                          self.debug_target_pose_w2[:, 3:]),), 
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
        # Obtains the pose of the base of the GEN3 robot in the world frame
        robot_root_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.root_state_w[:, 0:7]
        
        # --- Target pose ---
        # Obtain the pose for the target in the world frame
        tgt_pose_w = self.scene.rigid_objects["object"].data.body_state_w[:, :, :7].squeeze(-2)

        grasp_point_obj_pos_r, grasp_point_obj_quat_r = subtract_frame_transforms(t01 = robot_root_pose_w[:, :3], q01 = robot_root_pose_w[:, 3:],
                                                                                  t02 = tgt_pose_w[:, :3], q02 = tgt_pose_w[:, 3:])

        grasp_point_obj_pos_r_rot, grasp_point_obj_quat_r_rot = combine_frame_transforms(t01 = grasp_point_obj_pos_r, q01 = grasp_point_obj_quat_r,
                                                                                         t12 = self.z_displ, q12 = self.cfg.rot_45_z_pos_quat)
        

        tgt_pos_rot_w, tgt_rot_rot_w  = combine_frame_transforms(t01 = robot_root_pose_w[:, :3], q01 = robot_root_pose_w[:, 3:],
                                                                 t12 = grasp_point_obj_pos_r_rot, q12 = grasp_point_obj_quat_r_rot)
        

        self.debug_target_pose_w = torch.cat((tgt_pos_rot_w, tgt_rot_rot_w), dim = -1)

        self.target_pose_r = torch.cat((grasp_point_obj_pos_r_rot, grasp_point_obj_quat_r_rot), dim = -1)
        self.target_pose_r_group = self.convert_to_group(grasp_point_obj_pos_r_rot, grasp_point_obj_quat_r_rot)
        self.target_pose_r_lie = self.log(self.target_pose_r_group)

        # self.target_pose_r2 = torch.cat((grasp_point_obj_pos_r_rot, grasp_point_obj_quat_r_rot), dim = -1)
        self.target_pose_r_group2 = self.convert_to_group(self.target_pose_r2[:, :3], self.target_pose_r2[:, 3:])
        self.target_pose_r_lie2 = self.log(self.target_pose_r_group2)

        tgt_pos_rot_w2, tgt_rot_rot_w2  = combine_frame_transforms(t01 = robot_root_pose_w[:, :3], q01 = robot_root_pose_w[:, 3:],
                                                                 t12 = self.target_pose_r2[:, :3], q12 = self.target_pose_r2[:, 3:])
        
        self.debug_target_pose_w2 = torch.cat((tgt_pos_rot_w2, tgt_rot_rot_w2), dim = -1)


        # --- Robot poses ---
        # Obtain the pose of the GEN3 end effector in world frame
        self.debug_robot_ee_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.body_state_w[:, self.ee_jacobi_idx+1, 0:7]
        vel = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.body_state_w[:, self.ee_jacobi_idx+1, 7:]


        # Obtain the pose of the end effector in GEN3 root frame
        robot_rot_ee_pos_r, robot_rot_ee_quat_r = subtract_frame_transforms(t01 = robot_root_pose_w[:, :3], q01 = robot_root_pose_w[:, 3:],
                                                                              t02 = self.debug_robot_ee_pose_w[:, :3], q02 = self.debug_robot_ee_pose_w[:, 3:])

        # Build the group object
        self.pose_group_r = self.convert_to_group(robot_rot_ee_pos_r, robot_rot_ee_quat_r)

        self.target_pose_r = self.target_pose_r * torch.logical_not(self.target_reached).unsqueeze(-1) + self.target_pose_r2 * self.target_reached.unsqueeze(-1) 
        self.target_pose_r_group = self.target_pose_r_group * torch.logical_not(self.target_reached).unsqueeze(-1) + self.target_pose_r_group2 * self.target_reached.unsqueeze(-1)
        self.target_pose_r_lie = self.target_pose_r_lie * torch.logical_not(self.target_reached).unsqueeze(-1) + self.target_pose_r_lie2 * self.target_reached.unsqueeze(-1)

        # Transform to the Lie algebra
        self.robot_rot_ee_pose_r_lie = self.log(self.pose_group_r)
        diff = self.diff_operator(self.target_pose_r_group, self.pose_group_r)
        
        self.robot_rot_ee_pose_r_lie_rel = self.log(diff)



        self.obs_seq_vel_lie = update_seq(new_obs = vel, seq = self.obs_seq_vel_lie)
        self.obs_seq_pose_lie_rel = update_seq(new_obs = self.robot_rot_ee_pose_r_lie_rel, seq = self.obs_seq_pose_lie_rel)

        self.hand_joints_pos = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.joint_pos[:, self._hand_joints_idx]
        self.hand_pose = torch.round(self.hand_joints_pos[:, 2] / self.cfg.m[0], decimals = 0) / 140.0


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

        # Obtain boolean values for collisions
        self.filter_collisions()
        
        # Builds the tensor with all the observations in a single row tensor (N, 7+7+1)
        # obs = self.robot_rot_ee_pose_r_lie_rel
        obs = torch.cat((self.robot_rot_ee_pose_r_lie_rel, self.hand_pose.unsqueeze(-1)), dim = -1)
        # obs = self.obs_seq_pose_lie_rel.view(self.num_envs, -1)

        
        # obs = self.obs_seq_robot_pose_r_lie_rel.view(self.num_envs, -1)

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

        dist = self.dist_function(self.pose_group_r, self.target_pose_r_group, self.log, self.diff_operator)
        dist2 = self.dist_function(self.pose_group_r, self.target_pose_r_group2, self.log, self.diff_operator)   

        # --- Contacts ---
        contacts_w = (self.contacts * self.cfg.contact_matrix).sum(-1)
        is_contact = contacts_w > 4

        diff_actions = (2*(self.teacher_action == self.student_action) - 1).sum(-1) / 3        
        

        # Obtains wether the agent is approaching or not
        mod = (2*(dist < self.prev_dist).int() - 1).float()

        aux_reached = self.target_reached.clone()

        # Target reached flag
        self.target_reached = torch.logical_or(dist < self.cfg.distance_thres, self.target_reached)
        self.height_reached = torch.logical_and(dist2 < self.cfg.distance_thres, self.target_reached) # self.target_pose_r[:, 2] >= self.cfg.height_thres

        apply_bonus = torch.logical_and(torch.logical_not(aux_reached), self.target_reached)


        # ---- Distance reward ----
        # Reward for the approaching
        reward = diff_actions * torch.logical_or(torch.logical_not(self.target_reached), is_contact) #* torch.logical_not(self.target_reached) # mod * self.cfg.rew_scale_dist * torch.exp(-2*dist)


        # ---- Reward composition ----
        # Phase reward plus bonuses
        reward = reward + apply_bonus * self.cfg.bonus_tgt_reached + contacts_w

        # Reward for lifting
        reward = reward + is_contact * self.target_reached * torch.exp(-2*dist2) * self.cfg.bonus_lifting + self.height_reached * self.cfg.bonus_tgt_reached

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
        terminated = torch.logical_or(time_out, self.height_reached)

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


    def reset_target(self, env_ids, targets, ranges):
        target_pose_r = targets[0].clone()
        target_pose_r_group = targets[1].clone()
        target_pose_r_lie = targets[2].clone()

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
        target_pose_r_lie[env_ids] = self.log(self.target_pose_r_group)[env_ids]

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

        self.obs_seq_vel_lie[env_ids] = torch.zeros((self.num_envs, self.cfg.seq_len, 6)).to(self.device).float()[env_ids]

        # --- Reset controller ---
        self.controller.reset()
        
        # --- Reset target ---
        # Builds the new initial pose for the target
        self.target_pose_r, self.target_pose_r_group, self.target_pose_r_lie = self.reset_target(env_ids = env_ids,
                                                                                                 targets=(self.target_pose_r, 
                                                                                                          self.target_pose_r_group, 
                                                                                                          self.target_pose_r_lie),
                                                                                                 ranges = self.target_pose_ranges)

        grasp_point_obj_pos_r_rot, grasp_point_obj_quat_r_rot = combine_frame_transforms(t01 = self.target_pose_r[:, :3], q01 = self.target_pose_r[:, 3:],
                                                                                         t12 = 0*self.z_displ.to(self.device), q12 = self.cfg.rot_45_z_pos_quat)
        
        grasp_point_obj_pos_r_180, grasp_point_obj_quat_r_180 = combine_frame_transforms(t01 = grasp_point_obj_pos_r_rot, q01 = grasp_point_obj_quat_r_rot,
                                                                                          t12 = torch.zeros_like(grasp_point_obj_pos_r_rot), q12 = self.cfg.rot_180_z_pos_quat)
        
        target_pose_r = torch.cat((grasp_point_obj_pos_r_rot, grasp_point_obj_quat_r_rot), dim = -1)
        target_pose_r_group = self.convert_to_group(grasp_point_obj_pos_r_rot, grasp_point_obj_quat_r_rot)
        target_pose_r_lie = self.log(target_pose_r_group)

        target_pose_r_180 = torch.cat((grasp_point_obj_pos_r_180, grasp_point_obj_quat_r_180), dim = -1)
        target_pose_r_group_180 = self.convert_to_group(grasp_point_obj_pos_r_180, grasp_point_obj_quat_r_180)
        target_pose_r_lie_180 = self.log(target_pose_r_group_180)


        dist = self.dist_function(self.pose_group_r, target_pose_r_group, self.log, self.diff_operator)
        dist_180 = self.dist_function(self.pose_group_r, target_pose_r_group_180, self.log, self.diff_operator)    

        obs_rel = self.diff_operator(target_pose_r_group, self.pose_group_r)
        obs_rel_180 = self.diff_operator(target_pose_r_group_180, self.pose_group_r)

        sel_obs = (dist_180 > dist).int().unsqueeze(-1) 
        self.robot_rot_ee_pose_r_lie_rel = self.log(obs_rel * sel_obs + obs_rel_180 * torch.logical_not(sel_obs))

        self.target_pose_r[env_ids] = target_pose_r[env_ids] * sel_obs[env_ids] + target_pose_r_180[env_ids] * torch.logical_not(sel_obs)[env_ids]
        self.target_pose_r_group[env_ids] = target_pose_r_group[env_ids]  * sel_obs[env_ids] + target_pose_r_group_180[env_ids] * torch.logical_not(sel_obs)[env_ids]
        self.target_pose_r_lie[env_ids] = target_pose_r_lie[env_ids]  * sel_obs[env_ids] + target_pose_r_lie_180[env_ids] * torch.logical_not(sel_obs)[env_ids]






    
        self.target_pose_r2, self.target_pose_r_group2, self.target_pose_r_lie2 = self.reset_target(env_ids = env_ids,
                                                                                                    targets=(self.target_pose_r2, 
                                                                                                          self.target_pose_r_group2, 
                                                                                                          self.target_pose_r_lie2),
                                                                                                    ranges = self.target_pose_ranges2)

        # --- Reset previous values ---
        # Reset previous distances
        self.prev_dist[env_ids] = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)[env_ids]
        self.height_reached[env_ids] = torch.zeros(self.num_envs).bool().to(self.device)[env_ids]
        self.target_reached[env_ids] = torch.zeros(self.num_envs).bool().to(self.device)[env_ids]


        obs_rel = self.diff_operator(self.target_pose_r_group, self.pose_group_r)

        self.robot_rot_ee_pose_r_lie_rel[env_ids] = self.log(obs_rel)[env_ids]

        self.obs_seq_pose_lie_rel[env_ids] = torch.repeat_interleave(self.log(obs_rel), 
                                                                     self.cfg.seq_len, 
                                                                     dim=0).view(self.num_envs,self.cfg.seq_len,-1)[env_ids]
        


        robot_root_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.robot]].data.root_state_w[:, 0:7]
        tgt_pos_w, tgt_quat_w = combine_frame_transforms(t01 = robot_root_pose_w[:, :3], q01 = robot_root_pose_w[:, 3:],
                                                         t12 = self.target_pose_r[:, :3], q12 = self.target_pose_r[:, 3:])


        # Writes the new object position to the simulation
        self.scene.rigid_objects["object"].write_root_pose_to_sim(root_pose = torch.cat((tgt_pos_w, 
                                                                                         tgt_quat_w), dim = -1)[env_ids], env_ids = env_ids)
        self.scene.rigid_objects["object"].write_root_velocity_to_sim(root_velocity = torch.zeros((self.num_envs, 6), device=self.device)[env_ids], env_ids = env_ids)
        
        self.contacts[env_ids] = torch.empty(self.num_envs, self.num_contacts).fill_(False).to(self.device)[env_ids]

        # Updates the poses 
        self.update_new_poses()  




        # # Sets the command to the DifferentialIKController
        # self.controller.set_command(self.target_pose_r)

        # # Obtains current poses for the robot
        # ee_pos_r, ee_quat_r, jacobian, joint_pos = self._get_ee_pose()  

        # # Obtains the joint positions to reset. Concatenates:
        # #   - the joint coordinates for the action computed by the IKDifferentialController and
        # #   - the joint coordinates for the hand.
        # new_joint_pos = torch.cat((self.controller.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos), 
        #                        self.default_joint_pos[:, (6):]), 
        #                        dim=-1)

        # joint_pos = torch.cat((joint_pos, self.default_joint_pos[:, (6):]), dim=-1)

        # move_tgt = torch.rand((self.num_envs, 1)).to(self.device) < 0.5
        # self.target_reached[env_ids] = torch.zeros(self.num_envs).bool().to(self.device)[env_ids] * move_tgt.squeeze(-1)[env_ids] + torch.ones(self.num_envs).bool().to(self.device)[env_ids] * torch.logical_not(move_tgt.squeeze(-1))[env_ids]

        # joint_pos_ = joint_pos * move_tgt.int() + new_joint_pos * torch.logical_not(move_tgt).int()

        # # Writes the state to the simulation
        # self.scene.articulations[self.cfg.keys[self.cfg.robot]].write_joint_state_to_sim(joint_pos_[env_ids], self.default_vel[env_ids], None, env_ids)
        
        # new_joint_pos[:, 2+6] = 0.6        
        # new_joint_pos[:, 3+6] = 0.6
        # new_joint_pos[:, 4+6] = 0.6        
        # new_joint_pos[:, -1] = -0.6
        # new_joint_pos[:, -2] = -0.6        
        # new_joint_pos[:, -3] = -0.6

        # self.scene.articulations[self.cfg.keys[self.cfg.robot]].write_joint_state_to_sim(new_joint_pos, joint_vel, None, env_ids)
