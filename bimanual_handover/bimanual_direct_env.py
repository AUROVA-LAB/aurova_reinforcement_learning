from __future__ import annotations

import os
import torch
from collections.abc import Sequence

from .mdp.utils import compute_rewards, save_images_grid
from .bimanual_direct_env_cfg import BimanualDirectCfg, update_cfg

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, combine_frame_transforms
from omni.isaac.lab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat
from omni.isaac.lab.sensors import Camera, ContactSensorCfg, ContactSensor
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


class BimanualDirect(DirectRLEnv):
    cfg: BimanualDirectCfg

    def __init__(self, cfg: BimanualDirectCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # WILL BE REMOVED: initial poses sampled in reset for both robots
        self.new_poses = [[], []]

        # Indexes for: robot joints, hand joints, all joints
        self._robot_joints_idx = [self.scene.articulations[key].find_joints(self.cfg.joints[idx])[0] for idx, key in enumerate(self.cfg.keys)]
        self._hand_joints_idx = [self.scene.articulations[key].find_joints(self.cfg.hand_joints[idx])[0] for idx, key in enumerate(self.cfg.keys)]
        self._all_joints_idx = [self.scene.articulations[key].find_joints(self.cfg.all_joints[idx])[0] for idx, key in enumerate(self.cfg.keys)]


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
        self.ee_jacobi_idx = torch.tensor([self.scene.articulations[key].find_bodies(self.cfg.ee_link[idx])[0][0] - 1 for idx, key in enumerate(self.cfg.keys)]).to(self.device)
       

        # List for the default joint poses of both robots --> As a list due to the different joints of the arms (6 and 7) 
        self.default_joint_pos = [self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.default_joint_pos,
                                  self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.default_joint_pos]


        # Update configuration class
        self.cfg = update_cfg(cfg = cfg, num_envs = self.num_envs, device = self.device)

        # Obtain the ranges in which sample reset positions
        self.ee_pose_ranges = torch.tensor([[ [i + inc[0], i + inc[1]] for i, inc in zip(poses, cfg.ee_pose_incs)] for poses in cfg.ee_init_pose]).to(self.device)


        # Create output directory to save images
        self.output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.count = 0
    
    
    # Method to add all the prims to the scen --> Overrides method of DirectRLEnv
    def _setup_scene(self, ):
        '''
        NOTE: The "self.scene" variable is declared at "super().__init__(cfg, render_mode, **kwargs)" in __init__
        '''

        # Correct contact sensors cfg --> Now the number of environments is known
        self.cfg.contact_forces = ContactSensorCfg(
            prim_path="/World/envs/env_.*/" + self.cfg.keys[self.cfg.UR5e] + "/.*_tip", 
            update_period=0.1, 
            history_length=6, 
            debug_vis=True,
            filter_prim_paths_expr =[f"/World/envs/env_{i}/Cuboid" for i in range(self.num_envs)]
        )
        

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
        self.scene.sensors["camera"] = Camera(self.cfg.camera_cfg)
        self.scene.sensors["contact_sensors"] = ContactSensor(self.cfg.contact_forces)

        # Add bodies
        self.scene.rigid_objects["object"] = RigidObject(self.cfg.object_cfg)
        
        # Add extras (markers, ...)
        self.scene.extras["markers"] = VisualizationMarkers(self.cfg.marker_cfg)

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    # Method to preprocess the actions so they have a proper format
    def _preprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        '''
        In:
            - actions - torch.Tensor: raw actions. --> rotation is in the form of a quaternion.
                Format: [x, y, z, alpha, x_, y_, z_]:
                    actions[:3]: translation.
                    actions[3]: rotation angle of a quaternion.
                    actions[4:]: rotation vector of a quaternion

        Out:
            - actions - torch.Tenso: preprocessed actions.
        '''

        # Scale angle and rotation vector
        actions[:, 3] *= self.cfg.angle_scale
        actions[:, 4:7] = torch.nn.functional.normalize(actions[:, 4:7])

        # Real part of the quaternion
        w = torch.cos(actions[:, 3]/2).unsqueeze(dim = 0).T
        
        # Imaginary part of the quaternion
        v = actions[:, 4:7]
        sin_a = torch.sin(actions[:, 3] / 2).unsqueeze(dim=0).T

        # Build the quaternion
        q = sin_a * v

        # Reassign quaternion
        actions[:, 3:7] = torch.cat((w, q), dim = 1)
        
        return actions
    
    
    # Obtain the end effector pose of the index robot in the base frame
    def _get_ee_pose(self, idx):
        '''
        In: 
            - idx - int(0,1): index of the robot.

        Out:
            - ee_pos_b - torch.tensor(N, 3): position of the end effector in the base frame for each environment.
            - ee_quat_b - torch.tensor(N, 4): orientation as a quaternions of the end effector in the base frame for each environment.
            - jacobian - torch.tensor(N, 6, n_joints (6 or 7)): jacobian of all robots' end effector. 
            - joint_pos - torch.tensor(N, n_joints(6 or 7)): joint position of the robot.
        '''
        
        # Obtains the jacobian of the end effector of the robot
        jacobian = self.scene.articulations[self.cfg.keys[idx]].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx[idx], :, self._robot_joints_idx[idx]]

        # Obtains the pose of the end effector in the world frame
        ee_pose_w = self.scene.articulations[self.cfg.keys[idx]].data.body_state_w[:, self.ee_jacobi_idx[idx]+1, 0:7]

        # Obtains the pose of the  base of the robot in the world frame
        root_pose_w = self.scene.articulations[self.cfg.keys[idx]].data.root_state_w[:, 0:7]
        
        # Obtains the joint position
        joint_pos = self.scene.articulations[self.cfg.keys[idx]].data.joint_pos[:, self._robot_joints_idx[idx]]

        # Transforms end effector frame coordinates (in world) into root (local / base) coordinates
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
        # root = T01 // ee = T02 -> substract = (T01)^-1 * T02 = T10 * T02 = T12
        
        return ee_pos_b, ee_quat_b, jacobian, joint_pos


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
        self.controller.set_command(actions[:, :7])

        # Obtains the poses
        ee_pos_b, ee_quat_b, jacobian, joint_pos = self._get_ee_pose(self.cfg.UR5e)        
        
        # Get the actions for the UR5e. Concatenates:
        #   - the joint coordinates for the action computed by the IKDifferentialController and
        #   - the joint coordinates for the hand.
        self.actions_1 = torch.cat((self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos), 
                                    actions[:, 7:]), 
                                    dim = -1)


        # --- GEN3 actions ---
        # Set the command for the IKDifferentialController
        self.controller.set_command(actions[:,:7])

        # Obtains the poses
        ee_pos_b, ee_quat_b, jacobian, joint_pos = self._get_ee_pose(self.cfg.GEN3)
        
        # Get the actions for the UR5e. Concatenates:
        #   - the joint coordinates for the action computed by the IKDifferentialController and
        #   - the joint coordinates for the hand.
        self.actions_2 = torch.cat((self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos), 
                                    actions[:, 7:]), 
                                    dim = -1)


    # Applies the preprocessed action in the environment --> Overrides method of DirecRLEnv
    def _apply_action(self) -> None:
        
        # Applies joint actions to the robots 
        self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].set_joint_position_target(self.new_poses[self.cfg.UR5e], joint_ids=self._all_joints_idx[self.cfg.UR5e])
        self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].set_joint_position_target(self.new_poses[self.cfg.GEN3], joint_ids=self._all_joints_idx[self.cfg.GEN3])


    # Transforms the coordinates of the commands from base (local) to world frames
    def obtain_world_cmds(self, idx):
        '''
        In: 
            - idx - int(0 or 1): index for the robot.

        Out:
            - ee_pose_w[:, 0:3] - torch.tensor(N, 3): position of the end effector of the robot in world frame.
            - ee_pose_w[:, 3:] - torch.tensor(N,4): orientation as a quaternion of the end effector of the robot in world frame.
            - cmd_pos_w - torch.tensor(N, 3): position of the command of the robot in world frame.
            - cmd_quat_w- torch.tensor(N,4): orientation as a quaternion of the command of the robot in world frame.
        '''

        # Obtains the pose of the end effector in the world frame
        ee_pose_w = self.scene.articulations[self.cfg.keys[idx]].data.body_state_w[:, self.ee_jacobi_idx[idx]+1, 0:7]

        # Obtains the pose of the robot base in the world frame
        root_pose_w = self.scene.articulations[self.cfg.keys[idx]].data.root_state_w[:, 0:7]

        # Converts command pose in the base frame to the world frame
        cmd_pos_w, cmd_quat_w = combine_frame_transforms(t01 = root_pose_w[:, :3], q01 = root_pose_w[:, 3:],
                                                         t12 = self.obj_cmd[:, :3],    q12 = self.obj_cmd[:, 3:])
        # root = T01 // cmd = T13 -> add = T01 * T13 = T13

        return ee_pose_w[:, 0:3], ee_pose_w[:, 3:], cmd_pos_w, cmd_quat_w


    # Update the position of the markers with debug purposes
    def update_markers(self):
        '''
        Current markers:
            - End effector of the UR5e.
            - End effector of the GEN3.
            - Spawning position for the object.
        '''
        
        # Obtains the positions of the of the robots
        ee_pose_w_1 = self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.body_state_w[:, self.ee_jacobi_idx[self.cfg.UR5e]+1, 0:7]
        ee_pose_w_2 = self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w[:, self.ee_jacobi_idx[self.cfg.GEN3]+1, 0:7]
        
        # Obtains a tensor of indices (a tensor containing tensors from 0 to the number of markers)
        marker_indices = torch.arange(self.scene.extras["markers"].num_prototypes).repeat(self.num_envs)
        
        # Updates poses in simulation
        self.scene.extras["markers"].visualize(translations = torch.cat((self.obj_cmd[:, :3], ee_pose_w_1[:, :3], ee_pose_w_2[:, :3])), 
                                                orientations = torch.cat((self.obj_cmd[:, 3:], ee_pose_w_1[:, 3:], ee_pose_w_2[:, 3:])), 
                                                marker_indices=marker_indices)
        

    # Getter for the observations for the environment --> Overrides method of DirectRLEnv
    def _get_observations(self) -> dict:
        '''
        In:
            - None
        
        Out:
            - observations - dict: observations from the environment --> Needs to be with "policy" key. 
        '''

        # Count of the simulation steps
        self.count += 1

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
        
        # Obtain end effector poses for the robots
        ee_pos_b_1, ee_quat_b_1, _, _ = self._get_ee_pose(self.cfg.UR5e)
        ee_pos_b_2, ee_quat_b_2, _, _ = self._get_ee_pose(self.cfg.GEN3)

        # Obtains the joint positions for the hand
        hand_joint_pos_1 = self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.joint_pos[:, self._hand_joints_idx[self.cfg.UR5e]]
        hand_joint_pos_2 = self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.joint_pos[:, self._hand_joints_idx[self.cfg.GEN3]]

        # TODO: sacar pose del objeto

        # Builds the tensor with all the observations in a single row tensor (N, 7+16+7+16)
        obs = torch.cat(
            (
                torch.cat((ee_pos_b_1, ee_quat_b_1), dim = -1),
                hand_joint_pos_1,
                torch.cat((ee_pos_b_2, ee_quat_b_2), dim = -1),
                hand_joint_pos_2,
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
        
        # Computes reward according to the scaling values and poses (in utils)
        return compute_rewards(self.cfg.rew_position_tracking,
                               self.cfg.rew_position_tracking_fine_grained,
                               self.cfg.rew_orientation_tracking,
                               self.cfg.rew_dual_quaternion_error,
                               self.cfg.rew_action_rate,
                               self.cfg.rew_joint_vel,
                               self.device)
    

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

        truncated = torch.logical_or(out_of_bounds_1, out_of_bounds_2)
        terminated = time_out

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
        joint_pos = self.default_joint_pos[idx]
        joint_vel = self.scene.articulations[self.cfg.keys[idx]].data.default_joint_vel

        # Obtains the root (base) poses and velocities of the robot in the local frame
        default_root_state = self.scene.articulations[self.cfg.keys[idx]].data.default_root_state

        # Adds the position of the environment in world frame --> transforms the root position of the robot in local frame to world frame 
        default_root_state[:, :3] += self.scene.env_origins

        # Write the poses, velocities and joint positions to the environments
        self.scene.articulations[self.cfg.keys[idx]].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.scene.articulations[self.cfg.keys[idx]].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
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
        
        # Somehow, UR5e orientation needs a sign change
        if idx == self.cfg.UR5e: quat *= -1
        
        # Builds the new initial pose
        ee_init_pose = torch.cat((ee_init_pose[:, :3], quat), dim = -1)

        # Sets the command to the DifferentialIKController
        self.controller.set_command(ee_init_pose)

        # Obtains current poses for the robot
        ee_pos_b, ee_quat_b, jacobian, joint_pos = self._get_ee_pose(idx)  

        # Obtains the joint positions to reset. Concatenates:
        #   - the joint coordinates for the action computed by the IKDifferentialController and
        #   - the joint coordinates for the hand.
        self.new_poses[idx] = torch.cat((self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos), 
                                         self.default_joint_pos[idx][:, (6+idx):]), 
                                         dim=-1)
        joint_pos = torch.cat((self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos), 
                               self.default_joint_pos[idx][:, (6+idx):]), 
                               dim=-1)
        
        # Obtains the joint velocities
        joint_vel = self.scene.articulations[self.cfg.keys[idx]].data.default_joint_vel
       
        # Writes the state to the simulation
        self.scene.articulations[self.cfg.keys[idx]].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


    # Resetes the simulation --> Overrides method of DirectRLEnv
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
        
        # Updates the command of the object, i.e. the spawning position
        self.obj_cmd = torch.cat((obj_pos, obj_quat), dim = -1)
        
