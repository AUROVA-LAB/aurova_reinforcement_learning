from __future__ import annotations

import os
import torch
from collections.abc import Sequence
import copy

from .mdp.utils import compute_rewards, save_images_grid
from .mdp.rewards import dual_quaternion_error 
from .bimanual_direct_env_cfg import BimanualDirectCfg, update_cfg, update_collisions

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, combine_frame_transforms
from omni.isaac.lab.utils.math import quat_from_euler_xyz
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.assets import RigidObject

'''
                    ############## IMPORTANT #################
   The whole environment is build for two robots: the UR5e and Kinova GEN3-7dof.
   These two variables (cfg.UR5e and cfg.GEN3) serve as an abstraction to treat the robots during the episodes. In fact,
all the methods need an index to differentiate from which robot get the information.
   Also, data storage is performed using lists, not tensors because the joint space of the robots is
different from one another.
'''

# Class for the Bimanual Direct Environment
class BimanualDirect(DirectRLEnv):
    cfg: BimanualDirectCfg

    # --- init function ---
    def __init__(self, cfg: BimanualDirectCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initial poses sampled in reset for both robots
        self.reset_joint_positions = [torch.zeros((self.num_envs, 6+16)).to(self.device), torch.zeros((self.num_envs, 7+16)).to(self.device)]

        # Debug poses for the object and end effector of the GEN3 robot. These poses 
        # are used to draw the markers in the simulation
        self.debug_GEN3_ee_pose_w = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)
        self.debug_grasp_point_obj_pose_w = copy.deepcopy(self.debug_GEN3_ee_pose_w)
        self.debug_tips_pose_w = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)
        self.debug_tips_back_pose_w = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)

        # Poses for the object and GEN3 robot
        self.GEN3_rot_ee_pose_r = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)
        self.grasp_point_obj_pose_r = copy.deepcopy(self.GEN3_rot_ee_pose_r)
        self.tips_pose_r = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)
        self.tips_pose_r_back = torch.tensor([0,0,0, 1,0,0,0]).to(self.device).repeat(self.num_envs, 1)

        # Indexes for: robot joints, hand joints, all joints, finger tips, end effector's jacobian
        self._robot_joints_idx = [self.scene.articulations[key].find_joints(self.cfg.joints[idx])[0] for idx, key in enumerate(self.cfg.keys)]
        self._hand_joints_idx = [self.scene.articulations[key].find_joints(self.cfg.hand_joints[idx])[0] for idx, key in enumerate(self.cfg.keys)]
        self._all_joints_idx = [self.scene.articulations[key].find_joints(self.cfg.all_joints[idx])[0] for idx, key in enumerate(self.cfg.keys)]
        self.finger_tips = torch.tensor([self.scene.articulations[key].find_bodies(self.cfg.finger_tips[idx])[0] for idx, key in enumerate(self.cfg.keys)]).to(self.device)
        self.ee_jacobi_idx = torch.tensor([self.scene.articulations[key].find_bodies(self.cfg.ee_link[idx])[0][0] - 1 for idx, key in enumerate(self.cfg.keys)]).to(self.device)


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

        # List for the default joint poses of both robots --> As a list due to the different joints of the arms (6 and 7) 
        self.default_joint_pos = [self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.default_joint_pos,
                                  self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.default_joint_pos]
        
        # Default joints to open the hand
        self.open_hand_joints = torch.zeros((1, 16)).to(self.device)
        self.open_hand_joints[:, 1] = 0.263  # this value is the zero for the joint0 of the thumb

        # List of joint actions
        self.actions = copy.deepcopy(self.default_joint_pos)

        # Poses obtained at reset
        self.reset_robot_poses_r = [torch.zeros((self.num_envs, 7)).to(self.device), torch.zeros((self.num_envs, 7)).to(self.device)] 

        # Update configuration class
        self.cfg = update_cfg(cfg = cfg, num_envs = self.num_envs, device = self.device)

        # Obtain the ranges in which sample reset positions
        self.ee_pose_ranges = torch.tensor([[ [(i + cfg.apply_range[idx]*inc[0]), (i + cfg.apply_range[idx]*inc[1])] for i, inc in zip(poses, cfg.ee_pose_incs)] for idx, poses in enumerate(cfg.ee_init_pose)]).to(self.device)

        # Obtain the number of contact sensors per environment
        self.num_contacts = 0
        for __ in self.cfg.contact_sensors_dict:
            self.num_contacts += 1

        # Variable to store contacts between prims
        self.contacts = torch.empty(self.num_envs, self.num_contacts).fill_(False).to(self.device)

        # Create output directory to save images
        if self.cfg.save_imgs:
            self.output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
            os.makedirs(self.output_dir, exist_ok=True)

        # Previous distances
        self.prev_dist = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)
        self.prev_dist_target = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)
        
        # Object reached flags
        self.obj_reached = torch.zeros(self.num_envs).to(self.device).bool()
        self.obj_reached_target = torch.zeros(self.num_envs).to(self.device).bool()

        # Reward tensor for the second phase
        self.rew_2 = torch.ones(self.num_envs).to(self.device)

        self.back = torch.tensor([0.0, 0.05, -0.051, 1, 0, 0, 0]).to(self.device).repeat(self.num_envs, 1)
    
    
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
        self.scene.articulations[self.cfg.keys[self.cfg.UR5e]] = Articulation(self.cfg.robot_cfg_1)
        self.scene.articulations[self.cfg.keys[self.cfg.GEN3]] = Articulation(self.cfg.robot_cfg_2)

        # Add sensors (cameras, contact_sensors, ...)
        # self.scene.sensors["camera"] = Camera(self.cfg.camera_cfg)
        
        # Correct collision sensors 
        self.cfg = update_collisions(self.cfg, num_envs = self.num_envs)
        for idx, sensor_cfg in self.cfg.contact_sensors_dict.items():
            self.scene.sensors[idx] = ContactSensor(sensor_cfg)

        # Add bodies
        self.scene.rigid_objects["object"] = RigidObject(self.cfg.object_cfg)
        
        # Add extras (markers, ...)
        self.scene.extras["markers"] = VisualizationMarkers(self.cfg.marker_cfg)

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    # Obtain the end effector pose of the index robot in the base frame
    def _get_ee_pose(self, idx):
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
        jacobian = self.scene.articulations[self.cfg.keys[idx]].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx[idx], :, self._robot_joints_idx[idx]]

        # Obtains the pose of the end effector in the world frame
        ee_pose_w = self.scene.articulations[self.cfg.keys[idx]].data.body_state_w[:, self.ee_jacobi_idx[idx]+1, 0:7]

        # Obtains the pose of the base of the robot in the world frame
        root_pose_w = self.scene.articulations[self.cfg.keys[idx]].data.root_state_w[:, 0:7]
        
        # Obtains the joint position
        joint_pos = self.scene.articulations[self.cfg.keys[idx]].data.joint_pos[:, self._robot_joints_idx[idx]]

        # Transforms end effector frame coordinates (in world) into root (local / base) coordinates
        ee_pos_r, ee_quat_r = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
        # root = T01 // ee = T02 -> substract = (T01)^-1 * T02 = T10 * T02 = T12
        
        return ee_pos_r, ee_quat_r, jacobian, joint_pos


    # Method to preprocess the actions so they have a proper format
    def _preprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        '''
        In:
            - actions - torch.Tensor: raw actions. --> rotation is in the form of a quaternion.
                Format: [x, y, z, rotation]:
                    actions[:3]: translation.
                    actions[3]: rotation angle of a quaternion.
                    actions[4:]: rotation as a: (1) vector of a quaternion or (2) euler angles.

        Out:
            - actions - torch.Tensor: preprocessed actions.
        '''

        # Clamp actions
        actions = torch.clamp(actions, -1, 1)

        # Scale actions
        actions[:, :3]    *= self.cfg.translation_scale
        actions[:, 9:12]  *= self.cfg.translation_scale

        # Action in quaternion form
        actions_quat = torch.zeros((self.num_envs, 2,(7+16))).to(self.device)
        actions_quat[:, self.cfg.UR5e, :3] = actions[:, :3]
        actions_quat[:, self.cfg.GEN3, :3] = actions[:, 9:12]

        # Manipulation phase case, preprocess the actions for the hand
        if self.cfg.phase == self.cfg.MANIPULATION:

            # Determines the index of the hand joints
            hand_joint_index = 6 + int(not self.cfg.euler_flag)
            
            # Obtains extended action
            val = actions[:, hand_joint_index:hand_joint_index+3]# * self.cfg.hand_joint_scale)# .repeat_interleave(4, dim = -1)
            val_2 = actions[:, 9+hand_joint_index:hand_joint_index+3+9]# * self.cfg.hand_joint_scale)

            val = torch.cat((val, val_2), dim = -1).view(self.num_envs, -1, 3)# torch.cat((val, val_2), dim = 0)

            # Índices para intercalar filas
            indices = torch.arange(val.size(0)).view(2, -1).T.flatten()

            # Tensor con filas intercaladas
            tensor_intercalado = val[indices]
            
            # Assigns the values to the respective joints
            # actions_quat[:, self.cfg.UR5e, 7] = 0
            actions_quat[:, :, 8] = val[:, :, 0] * 5
            # actions_quat[:, :, 9] = 0
            # actions_quat[:, :, 10] = 0
            actions_quat[:, :, 11] = val[:, :, 0] * 5
            # actions_quat[:, :, 12] = 0.0
            actions_quat[:, :, 13] = val[:, :, 0] * 5
            actions_quat[:, :, 14] = val[:, :, 0] * 5

            actions_quat[:, :, 15:] = tensor_intercalado.view(self.num_envs, 2, 3)[:, :, 1:].repeat_interleave(4, dim = -1)*3


        # If the actions are in euler, transform them to quaternion
        if self.cfg.euler_flag:
            actions[:, 3:6] *= self.cfg.angle_scale
            
            actions_quat[:, self.cfg.UR5e, 3:7] = quat_from_euler_xyz(roll = actions[:, 3],
                                                    pitch = actions[:, 4],
                                                    yaw = actions[:, 5])
            
            actions[:, 3+9:6+9] *= self.cfg.angle_scale
            
            actions_quat[:, self.cfg.GEN3, 3:7] = quat_from_euler_xyz(roll = actions[:, 3+9],
                                                    pitch = actions[:, 4+9],
                                                    yaw = actions[:, 5+9])

        # Else, the actions are already in quaternion form
        else:
            # Scale angle and rotation vector
            actions_quat[:, self.cfg.UR5e, 3] *= self.cfg.angle_scale
            actions_quat[:, self.cfg.UR5e, 4:7] = torch.nn.functional.normalize(actions_quat[:, 4:7])

            # Real part of the quaternion
            w = torch.cos(actions_quat[:, self.cfg.UR5e, 3]/2).unsqueeze(dim = 0).T
            
            # Imaginary part of the quaternion
            v = actions_quat[:, self.cfg.UR5e, 4:7]
            sin_a = torch.sin(actions_quat[:, self.cfg.UR5e, 3] / 2).unsqueeze(dim=0).T

            # Build the quaternion
            q = sin_a * v

            # Reassign quaternion
            actions_quat[:, self.cfg.UR5e, 3:7] = torch.cat((w, q), dim = 1)     


            # Scale angle and rotation vector
            actions_quat[:, self.cfg.GEN3, 3] *= self.cfg.angle_scale
            actions_quat[:, self.cfg.GEN3, 4:7] = torch.nn.functional.normalize(actions_quat[:, self.cfg.GEN3, 4:7])

            # Real part of the quaternion
            w = torch.cos(actions_quat[:, self.cfg.GEN3, 3]/2).unsqueeze(dim = 0).T
            
            # Imaginary part of the quaternion
            v = actions_quat[:, self.cfg.GEN3, 4:7]
            sin_a = torch.sin(actions_quat[:, self.cfg.GEN3, 3] / 2).unsqueeze(dim=0).T

            # Build the quaternion
            q = sin_a * v

            # Reassign quaternion
            actions_quat[:, self.cfg.GEN3, 3:7] = torch.cat((w, q), dim = 1)

        return actions_quat
    

    # Performs the action increment
    def perform_increment(self, idx, actions):
        '''
        In: 
            - idx - int(0,1): index of the robot.
            - actions - torch.tensor(N, 7 + 16): the increment to be performed to the actual pose and hand joint position.

        Out:
            - None
        '''

        # Obtains the poses
        ee_pos_r, ee_quat_r, jacobian, joint_pos = self._get_ee_pose(idx)

        # Perform an increment on the robot end effector in the root frame
        new_act_pos, new_act_quat = combine_frame_transforms(t01 = ee_pos_r, q01 = ee_quat_r,
                                                             t12 = actions[:, idx, 0:3], q12 = actions[:, idx, 3:7])
        new_poses = torch.cat((new_act_pos, new_act_quat), dim = -1)


        # Set the command for the IKDifferentialController
        self.controller.set_command(new_poses)

        # Perform the increment for the hand
        new_hand_joint_pos = self.scene.articulations[self.cfg.keys[idx]].data.joint_pos[:, self._hand_joints_idx[idx]] + actions[:, idx, 7:] * self.cfg.phase
        
        if idx == self.cfg.GEN3: new_hand_joint_pos[:, 5] = -0.65

        # Get the actions for the UR5e. Concatenates:
        #   - the joint coordinates for the action computed by the IKDifferentialController and
        #   - the joint coordinates for the hand.
        self.actions[idx] = torch.cat((self.controller.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos), 
                                       new_hand_joint_pos), 
                                       dim = -1)


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

        # --- UR5e actions ---
        # Set the command for the IKDifferentialController
        self.perform_increment(idx = self.cfg.UR5e, actions = actions)

        # --- GEN3 actions ---
        # Obtains the increments and the poses for the GEN3 robot
        self.perform_increment(idx = self.cfg.GEN3, actions = actions)


    # Applies the preprocessed action in the environment --> Overrides method of DirecRLEnv
    def _apply_action(self) -> None:
        
        # Applies joint actions to the robots 
        self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].set_joint_position_target(self.actions[self.cfg.UR5e], joint_ids=self._all_joints_idx[self.cfg.UR5e])
        self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].set_joint_position_target(self.actions[self.cfg.GEN3], joint_ids=self._all_joints_idx[self.cfg.GEN3])


    # Update the position of the markers with debug purposes
    def update_markers(self):
        '''
        Current markers:
            - End effector of the UR5e.
            - End effector of the GEN3.
            - Grasping position for the object (transformated to match GEN3's).
            - Tips of the fingers of the GEN3.
            - Tips of the fingers of the GEN3 (displaced in front of the hand).
        '''
        
        # Obtains the positions of the of the robots
        ee_pose_w_UR5e = self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.body_state_w[:, self.ee_jacobi_idx[self.cfg.UR5e]+1, 0:7]

        # Obtains a tensor of indices (a tensor containing tensors from 0 to the number of markers)
        marker_indices = torch.arange(self.scene.extras["markers"].num_prototypes).repeat(self.num_envs)

        # Updates poses in simulation
        self.scene.extras["markers"].visualize(translations = torch.cat((ee_pose_w_UR5e[:, :3], 
                                                                         self.debug_GEN3_ee_pose_w[:, :3],
                                                                         self.debug_grasp_point_obj_pose_w[:, :3],
                                                                         self.debug_tips_pose_w[:, :3], 
                                                                         self.debug_tips_back_pose_w[:, :3]),), 
                                                orientations = torch.cat((ee_pose_w_UR5e[:, 3:], 
                                                                          self.debug_GEN3_ee_pose_w[:, 3:],
                                                                          self.debug_grasp_point_obj_pose_w[:,3:],
                                                                          self.debug_tips_pose_w[:, 3:],
                                                                          self.debug_tips_back_pose_w[:, 3:]),), 
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


    # Updates the poses of the object and robots so they can match when performing the grasp
    def update_new_poses(self):
        '''
        In:
            - None
        
        Out:
            - None
        '''

        # ---- GEN3 transformations ----
        # Obtain the pose of the GEN3 end effector in world frame
        self.debug_GEN3_ee_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w[:, self.ee_jacobi_idx[self.cfg.GEN3]+1, 0:7]

        # Obtains the pose of the base of the GEN3 robot in the world frame
        GEN3_root_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.root_state_w[:, 0:7]

        # Obtain the pose of the end effector in GEN3 root frame
        GEN3_rot_ee_pos_r, GEN3_rot_ee_quat_r = subtract_frame_transforms(t01 = GEN3_root_pose_w[:, :3], q01 = GEN3_root_pose_w[:, 3:],
                                                                              t02 = self.debug_GEN3_ee_pose_w[:, :3], q02 = self.debug_GEN3_ee_pose_w[:, 3:])

        self.GEN3_rot_ee_pose_r = torch.cat((GEN3_rot_ee_pos_r, GEN3_rot_ee_quat_r), dim = -1)




        # ---- Tips transformations ----
        # Obtains the pose of the finger tips in world frame and performs the mean
        self.debug_tips_pose_w = torch.mean(self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w[:, self.finger_tips[self.cfg.GEN3], 0:7], dim = -2)
        
        # Transform tips pose to GEN3 root frame
        tip_pos_r, tip_or_r = subtract_frame_transforms(t01 = GEN3_root_pose_w[:, :3], q01 = GEN3_root_pose_w[:, 3:],
                                                        t02 = self.debug_tips_pose_w[:, :3], q02 = self.debug_tips_pose_w[:, 3:])
        self.tips_pose_r = torch.cat((tip_pos_r, tip_or_r), dim = -1)

        # Replaces the orientation with GEN3 ee orientation
        self.tips_pose_r[:, 3:] = self.GEN3_rot_ee_pose_r[:, 3:]




        # Clones the tips original pose
        self.tips_pose_r_back = self.tips_pose_r.clone()

        # Displaces the tips pose in front of the hand
        tip_pos_r, tip_or_r = combine_frame_transforms(t01 = self.tips_pose_r[:, :3], q01 = self.tips_pose_r[:, 3:],
                                                        t12 = self.cfg.tips_displacement, q12 = torch.tensor([1,0,0,0]).to(self.device).repeat(self.num_envs, 1))
        self.tips_pose_r = torch.cat((tip_pos_r, tip_or_r), dim = -1)




        # Transforms the modified tips pose to the world frame
        tips_pos_w, tips_or_w = combine_frame_transforms(t01 = GEN3_root_pose_w[:, :3], q01 = GEN3_root_pose_w[:, 3:],
                                                         t12 = self.tips_pose_r[:, :3], q12 = self.tips_pose_r[:, 3:])
        self.debug_tips_pose_w = torch.cat((tips_pos_w, tips_or_w), dim = -1)
        
        # Transforms the modified tips back pose to the world frame
        tips_pos_w, tips_or_w = combine_frame_transforms(t01 = GEN3_root_pose_w[:, :3], q01 = GEN3_root_pose_w[:, 3:],
                                                         t12 = self.tips_pose_r_back[:, :3], q12 = self.tips_pose_r_back[:, 3:])
        self.debug_tips_back_pose_w = torch.cat((tips_pos_w, tips_or_w), dim = -1)




        # ---- Object transformations ----
        # Obtains the pose of the object in the world frame
        obj_pose_w = self.scene.rigid_objects["object"].data.body_state_w[:, 0, :7]

        # Transforms the object frame so as to generate a more suitable frame for grasping
        grasp_point_obj_pos_w, grasp_point_obj_quat_w = combine_frame_transforms(t01 = obj_pose_w[:, :3], q01 = obj_pose_w[:, 3:],
                                                                             t12 = self.cfg.grasp_obs_obj_pos_trans, q12 = self.cfg.grasp_obs_obj_quat_trans)
        grasp_point_obj_pos_w, grasp_point_obj_quat_w = combine_frame_transforms(t01 = grasp_point_obj_pos_w, q01 = grasp_point_obj_quat_w,
                                                                             t12 = torch.zeros_like(grasp_point_obj_pos_w), q12 = self.cfg.rot_225_z_pos_quat)
        self.debug_grasp_point_obj_pose_w = torch.cat((grasp_point_obj_pos_w, grasp_point_obj_quat_w), dim=-1)

        # Apply transformation to get the grasping point in the GEN3 root frame
        grasp_point_obj_pos_r, grasp_point_obj_quat_r = subtract_frame_transforms(t01 = GEN3_root_pose_w[:, :3], q01 = GEN3_root_pose_w[:, 3:],
                                                                              t02 = grasp_point_obj_pos_w, q02 = grasp_point_obj_quat_w)
        self.grasp_point_obj_pose_r = torch.cat((grasp_point_obj_pos_r, grasp_point_obj_quat_r), dim = -1)


    # Getter for the observations for the environment --> Overrides method of DirectRLEnv
    def _get_observations(self) -> dict:
        '''
        In:
            - None
        
        Out:
            - observations - dict: observations from the environment --> Needs to be with "policy" key. 
        '''

        # Obtain boolean values for collisions
        self.filter_collisions()

        # Updates the poses of the GEN3 end effector and the object so they match
        self.update_new_poses()

        # Render images every certain amount of steps
        if self.count % self.cfg.render_steps == 0 and self.cfg.render_imgs:
            
            # Obtain images from the sensor
            image_tensor = [self.scene["camera"].data.output["rgb"][0, ..., :3]]

            # Function to save images (in utils)
            if self.cfg.save_imgs:
                save_images_grid(images = image_tensor,
                                 subtitles = ["Camera"],
                                 title = "RGB Image: Cam0",
                                 filename = os.path.join(self.output_dir, "rgb", f"{self.count:04d}.jpg"))

        # Builds the tensor with all the observations in a single row tensor (N, 7+16)
        obs = torch.cat(
            (
                self.GEN3_rot_ee_pose_r,
                self.grasp_point_obj_pose_r,
            ),
            dim = -1
        )

        # In case of manipulation phase, add the hand joint positions
        if self.cfg.phase == self.cfg.MANIPULATION:

            # Obtains the joint positions for the hand
            hand_joint_pos_1 = self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.joint_pos[:, self._hand_joints_idx[self.cfg.UR5e]]
            hand_joint_pos_2 = self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.joint_pos[:, self._hand_joints_idx[self.cfg.GEN3]]
            
            # Selects the hand joints to be observed
            sel_hand_joint_1 = torch.round(torch.cat((hand_joint_pos_1[:, 1].unsqueeze(-1), hand_joint_pos_1[:, 4].unsqueeze(-1), hand_joint_pos_1[:, 6:]), dim = -1), decimals = 1)
            sel_hand_joint_2 = torch.round(torch.cat((hand_joint_pos_2[:, 1].unsqueeze(-1), hand_joint_pos_2[:, 4].unsqueeze(-1), hand_joint_pos_2[:, 6:]), dim = -1), decimals = 1)
            
            # Concatenates the mean of selected hand joint positions
            obs = torch.cat(
                (
                    obs,
                    sel_hand_joint_1.view(-1, 3,4).mean(dim = -1).view(-1, 3),
                    sel_hand_joint_2.view(-1, 3,4).mean(dim = -1).view(-1, 3),
                ),
                dim = -1
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


        # ---- Variable assignments ----
        rew_scale_hand_obj = self.cfg.rew_scale_hand_obj * torch.ones(self.num_envs).to(self.device)
        rew_scale_obj_target = self.cfg.rew_scale_obj_target
        ee_pose = self.tips_pose_r
        ee_pose_bask = self.tips_pose_r_back
        obj_pose = self.grasp_point_obj_pose_r
        prev_dist = self.prev_dist
        prev_dist_target = self.prev_dist_target
        obj_reach_target_thres = self.cfg.obj_reach_target_thres
        device = self.device
        target_pose = self.cfg.target_pose
        

        # ---- Distance computation ----
        # Dual quaternion distance between GEN3 hand tips and object
        hand_obj_dist = dual_quaternion_error(ee_pose, obj_pose, device)
        hand_obj_dist_back = dual_quaternion_error(ee_pose_bask, obj_pose, device)
        
        # Dual quaternion distance between object and target pose
        obj_target_dist = dual_quaternion_error(obj_pose, target_pose, device)


        # ---- Contact computation ----
        # Obtain the weighted contacts
        contacts_w = self.contacts * self.cfg.contact_matrix

        # Thumb contact
        thumb_w = contacts_w[:, -9:-4].clone()
        thumb_con = thumb_w.sum(-1) > 0.0        


        # ---- Flag ----
        # There is contact if the thumb and the fingers (finger collide without the thumb) are in contact
        contacts_flag = torch.logical_and(contacts_w[:, :-2].sum(-1) - thumb_w.sum(-1) > 0.4, thumb_con)

        # Reached flag pre-conditions
        bonus = self.obj_reached.clone().bool()

        # Check it the object is reached (contact with the object) and set it
        self.obj_reached = torch.logical_or(contacts_flag * (self.cfg.phase == self.cfg.MANIPULATION), self.obj_reached)
        
        # Reached flag after conditions
        new_bonus = self.obj_reached.clone().bool()

        # The bonus must be activated if the object is reached at this step
        bonus = torch.logical_and(new_bonus, torch.logical_not(bonus))

        # Check if the object has reached the target
        self.obj_reached_target = (obj_pose[:, 0] < 0.05).bool() # (obj_target_dist[:, 1] < obj_reach_target_thres).bool()


        # ---- Distance evaluation ----
        # Obtains the distance according to the object reached flag
        dist = hand_obj_dist[:, 0] * torch.logical_not(self.obj_reached).int() + obj_target_dist[:, 0] * self.obj_reached.int()
        prev_dist = prev_dist * torch.logical_not(self.obj_reached).int() + prev_dist_target * self.obj_reached.int()

        # Obtains wether the agent is approaching or not
        mod = torch.logical_and(dist < prev_dist, hand_obj_dist_back[:,0] > hand_obj_dist[:,0])
        mod = 2*(torch.logical_and(mod, contacts_w[:, -1] > 0.0)) - 1

        # Modifies scalation according to the contacts detected
        rew_scale_hand_obj = rew_scale_hand_obj / (self.contacts[:, 1:-2].sum(-1) + 1)        


        # ---- Distance reward ----
        # Reward for the first phase --> Approaching (mod) hand-obj distance divided by wether the object is approaching with the palm
        reward_1 = mod * rew_scale_hand_obj * torch.exp(-2*hand_obj_dist[:, 0]) / (1 + 2*(hand_obj_dist_back[:,0] < hand_obj_dist[:,0]).int())
        
        # Reward for the second phase --> Object-target distance the target
        reward_2 = rew_scale_obj_target * torch.exp(-2*obj_target_dist[:, 0])


        # ---- Reward composition ----
        # Phase reward plus phase 1 bonuses
        reward = (reward_1) * torch.logical_not(self.obj_reached) + 10*(reward_2 - contacts_w[:, -1]) * self.obj_reached + self.cfg.bonus_obj_reach * bonus / 2

        # Reward for the contacts
        reward = reward + contacts_w[:, 1:-2].sum(-1) 

        # Reward for reaching target
        reward = reward + self.cfg.bonus_obj_reach * self.obj_reached_target * (contacts_w[:, 1:-1].sum(-1) > 0.0).int()


        # Update previous distances
        self.prev_dist = hand_obj_dist[:, 0]
        self.prev_dist_target = obj_target_dist[:, 0]
            
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
        out_of_bounds_1 = torch.norm(self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.body_state_w[:, self.ee_jacobi_idx[self.cfg.UR5e]+1, 7:], dim = -1) > self.cfg.velocity_limit 
        out_of_bounds_2 = torch.norm(self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w[:, self.ee_jacobi_idx[self.cfg.GEN3]+1, 7:], dim = -1) > self.cfg.velocity_limit
        out_of_bounds = torch.logical_or(out_of_bounds_1, out_of_bounds_2)

        # Falling conditions
        GEN3_falling = self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w[:, self.ee_jacobi_idx[self.cfg.GEN3]+1, 2] < self.cfg.gen3_height_limit
        object_falling = self.scene.rigid_objects["object"].data.body_state_w[:, 0, 2] < self.cfg.object_height_limit
        falling = torch.logical_or(GEN3_falling, object_falling)
        
        # Contact conditions
        GEN3_ground_contact = self.contacts[:, 0]

        # Truncated and terminated variables
        truncated = torch.logical_or(torch.logical_or(falling, out_of_bounds), GEN3_ground_contact)
        terminated = torch.logical_or(time_out, torch.logical_and(self.obj_reached, self.obj_reached_target) * (self.cfg.phase == self.cfg.MANIPULATION) + self.obj_reached * (self.cfg.phase == self.cfg.APPROACH))


        return truncated, terminated
    
    
    # Resets the index robot JOINT positions
    def reset_robot(self, idx, env_ids):
        '''
        In:
            - idx - int(0 or 1): index for the robot.
            - env_ids - torch.tensor(m): IDs for the 'm' environments that need to be resetted.
        
        Out:
            - None
        '''

        # Default joint position for the robots
        joint_pos = self.default_joint_pos[idx][env_ids]
        joint_vel = self.scene.articulations[self.cfg.keys[idx]].data.default_joint_vel[env_ids]

        # Write the joint positions to the environments
        self.scene.articulations[self.cfg.keys[idx]].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


    # Resets the index robot according to their END EFFECTOR
    def reset_robot_ee(self, idx, env_ids):
        '''
        In:
            - idx - int(0 or 1): index for the robot.
            - env_ids - torch.tensor(m): IDs for the 'm' environments that need to be resetted.
        
        Out:
            - None
        '''

        # Sample a random position using the end effector ranges with the shape of all environmnets
        ee_init_pose = sample_uniform(
            self.ee_pose_ranges[idx, :, 0],
            self.ee_pose_ranges[idx, :, 1],
            [self.num_envs, self.ee_pose_ranges[idx, :, 0].shape[0]],
            self.device,
        )

        # Transforms Euler to quaternion
        quat = quat_from_euler_xyz(roll = ee_init_pose[:, 3],
                                    pitch = ee_init_pose[:, 4],
                                    yaw = ee_init_pose[:, 5])
        
        # Builds the new initial pose
        ee_init_pose = torch.cat((ee_init_pose[:, :3], quat), dim = -1)

        # Save sampled pose
        self.reset_robot_poses_r[idx][env_ids] = ee_init_pose[env_ids]

        # Sets the command to the DifferentialIKController
        self.controller.set_command(ee_init_pose)

        # Obtains current poses for the robot
        ee_pos_r, ee_quat_r, jacobian, joint_pos = self._get_ee_pose(idx)  

        # Obtains the joint positions to reset. Concatenates:
        #   - the joint coordinates for the action computed by the IKDifferentialController and
        #   - the joint coordinates for the hand.
        self.reset_joint_positions[idx][env_ids] = torch.cat((self.controller.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos), 
                                         self.default_joint_pos[idx][:, (6+idx):]), 
                                         dim=-1)[env_ids]
        joint_pos = torch.cat((self.controller.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos), 
                               self.default_joint_pos[idx][:, (6+idx):]), 
                               dim=-1)[env_ids] 
        
        # Obtains the joint velocities
        joint_vel = self.scene.articulations[self.cfg.keys[idx]].data.default_joint_vel[env_ids]
       
        # Writes the state to the simulation
        self.scene.articulations[self.cfg.keys[idx]].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


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

        # --- Reset the robots ---
        # Reset the robots first to the default joint position so the IK is easier to compute after
        self.reset_robot(idx = self.cfg.UR5e, env_ids = env_ids)
        self.reset_robot(idx = self.cfg.GEN3, env_ids = env_ids)

        # Reset the robot to a random Euclidean position
        self.reset_robot_ee(idx = self.cfg.UR5e, env_ids = env_ids)
        self.reset_robot_ee(idx = self.cfg.GEN3, env_ids = env_ids)

        # --- Reset controller ---
        self.controller.reset()
        
        # --- Reset object ---
        # Obtains the end effector position for the UR5e        
        ee_pose_w = self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.body_state_w[env_ids, self.ee_jacobi_idx[self.cfg.UR5e]+1, 0:7]

        # Transforms the translation and orientation of the object pose (in the end effector frame) to the world frame
        obj_pos, obj_quat = combine_frame_transforms(t01 = ee_pose_w[:, :3], q01 = ee_pose_w[:, 3:],
                                                     t12 = self.cfg.obj_pos_trans[env_ids], q12 = self.cfg.obj_quat_trans[env_ids])

        # Writes the new object position to the simulation
        self.scene.rigid_objects["object"].write_root_pose_to_sim(root_pose = torch.cat((obj_pos, obj_quat), dim = -1), env_ids = env_ids)

        # Updates the poses of the GEN3 end effector and the object in the reset
        self.update_new_poses()

        # Reset previous distances
        self.prev_dist[env_ids] = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)[env_ids]
        self.prev_dist_target[env_ids] = torch.tensor(torch.inf).repeat(self.num_envs).to(self.device)[env_ids]
        self.obj_reached[env_ids] = torch.zeros(self.num_envs).bool().to(self.device)[env_ids]
        self.obj_reached_target[env_ids] = torch.zeros(self.num_envs).bool().to(self.device)[env_ids]

        self.contacts[env_ids] = torch.empty(self.num_envs, self.num_contacts).fill_(False).to(self.device)[env_ids]

        