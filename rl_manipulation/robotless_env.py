import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import matplotlib.pyplot as plt



from utils_robotless import * 
from math_robotless import *



class BasicPoseEnv(gym.Env):
    """
    Basic RL environment for pose reaching.

    At reset:
        - Random initial pose is generated.
        - Random goal pose is generated.

    At every step:
        action = [dx, dy, dz, drx, dry, drz]

        pose_{t+1} = pose_t + action

    Observation:
        [current_pose, goal_pose]

    Action:
        6D pose increment.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pos_low=(-0.7, -0.1, 0.2),
        pos_high=(-0.3, 0.3, 0.7),
        rot_low=(2*np.pi/5, 7*np.pi/5, 0.78),
        rot_high=(8*np.pi/5, 13*np.pi/5, 3.927),
        action_scale=0.05,
        max_steps=100,
        position_weight=1/0.602,
        rotation_weight=1/0.988,
        goal_threshold=0.01,
    ):
        super().__init__()
        
        # Assign the functions according to configuration
        self.exp = map_list[representation][mapping][0]                 # Exponential mapping
        self.log = map_list[representation][mapping][1]                 # Logarithmic mapping
        self.convert_to_Lab = conversions[representation][0]                # Conversion Lie group to IsaacLab representation
        self.convert_to_group = conversions[representation][1]              # Conversion IsaacLab representation to Lie group     
        self.interpolator = interpolators[representation]                   # Interpolator function
        self.dist_function = distances[representation][distance]        # Distance function
        self.diff_operator = diff_operators[representation]                 # Difference operator
        self.mul_operator = mul_operators[representation]                   # Multiply operator
        self.normalize = normalizes[representation]                         # Normalization function
        self.action_scaling = scalings[representation][mapping]

        self.exp_dist = map_list[MAT][1][0]
        self.log_dist = map_list[MAT][1][1]
        
        self.to_group_dist = conversions[MAT][1]

        self.pos_low = np.asarray(pos_low, dtype=np.float32)
        self.pos_high = np.asarray(pos_high, dtype=np.float32)

        self.rot_low = np.asarray(rot_low, dtype=np.float32)
        self.rot_high = np.asarray(rot_high, dtype=np.float32)

        self.action_scale_rot = action_scaling[0]
        self.action_scale_pos = action_scaling[1]
        self.max_steps = max_steps

        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
        self.goal_threshold = goal_threshold

        # --------------------------------------------------
        # Action
        # --------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(size,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Observation
        # --------------------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(size * (int(not symmetry) + 1),),
            dtype=np.float32,
        )

        self.pose = np.zeros(6, dtype=np.float32)
        self.pose_group = np.zeros(size_group, dtype=np.float32)
        self.pose_map = np.zeros(size, dtype=np.float32)

        self.goal = np.zeros(6, dtype=np.float32)
        self.goal_group = np.zeros(size_group, dtype=np.float32)
        self.goal_map = np.zeros(size, dtype=np.float32)

        self.action = np.zeros(size, dtype=np.float32)

        self.step_count = 0
        self.end_episode = 100

        self.rot_max = 0
        self.pos_max = 0

        self.rot_min = 100000000
        self.pos_min = 100000000


    def sample_pose(self):
        """Generate a random pose."""

        return sample_random_pose(
            pos_low=self.pos_low,
            pos_high=self.pos_high,
            rot_low=self.rot_low,
            rot_high=self.rot_high,
        )

    def get_observation(self):
        """Build the observation."""

        pose_group = torch.tensor(self.pose_group).unsqueeze(0) 
        goal_group = torch.tensor(self.goal_group).unsqueeze(0)

        pose_map = torch.tensor(self.pose_map).unsqueeze(0) 
        goal_map = torch.tensor(self.goal_map).unsqueeze(0)

        if symmetry:
            
            diff = self.diff_operator(goal_group, pose_group)

            if mapping != 0:
                log_diff = self.log(diff)
                log_diff[:, :3] /= self.action_scale_rot              
                log_diff[:, 3:] /= self.action_scale_pos          

                self.rot_max = max(torch.max(torch.round(log_diff[:,:3], decimals=3)).item(), self.rot_max)
                self.pos_max = max(torch.max(torch.round(log_diff[:,3:], decimals=3)).item(), self.pos_max)

                self.rot_min = min(torch.min(torch.round(log_diff[:,:3], decimals=3)).item(), self.rot_min)
                self.pos_min = min(torch.min(torch.round(log_diff[:,3:], decimals=3)).item(), self.pos_min)  

                # print("POS MIN MAX: [", self.pos_min, self.pos_max, "]")
                # print("ROT MIN MAX: [", self.rot_min, self.rot_max, "]")
                
                # print("log diff: ", log_diff)
                # print("Lab Pose: ", self.pose)                
                # print("--------")

                return log_diff.squeeze(0).cpu().numpy()

            else:
                return diff

        else:
            if mapping != 0:
                return np.concatenate(
                        [
                            pose_map,
                            goal_map,
                        ]
                    ).astype(np.float32)
            else:
                
                return np.concatenate(
                        [
                            pose_group,
                            goal_group,
                        ]
                    ).astype(np.float32)

    def compute_distance(self):
        """Return position and rotation distance."""

        return pose_distance(
            self.pose,
            self.goal,
            self.log_dist,
            self.to_group_dist,
            self.rotation_weight,
            self.position_weight
        )

    def compute_reward(self):
        """Compute reward."""

        return compute_reward(
            self.pose,
            self.goal,
            self.log_dist,
            self.to_group_dist,
            self.action,
            position_weight=self.position_weight,
            rotation_weight=self.rotation_weight,
        )

    def check_goal(self):
        """
        Check whether the goal has been reached.
        """

        position_distance, rotation_distance = self.compute_distance()
        
        return (
            position_distance < self.goal_threshold
            and rotation_distance < self.goal_threshold 
        )

    def reset(self, *, seed=None, options=None):
        """
        Reset environment.

        A new random initial pose and goal are generated.
        """

        super().reset(seed=seed)

        self.step_count = 0
        self.pose_map_history = []

        # Random initial pose
        self.pose = self.sample_pose()
        self.pose_map, self.pose_group = convert_np_euler_2_map(self.pose, self.log, self.convert_to_group)

        # Random goal
        self.goal = self.sample_pose()
        self.goal_map, self.goal_group = convert_np_euler_2_map(self.goal, self.log, self.convert_to_group)

        observation = self.get_observation()

        self.pose_map_history = [self.pose_map]

        info = {
            "pose": self.pose,
            "goal": self.goal,
            "pose_map": self.pose_map,
            "goal_map": self.goal_map,
            "pose_group": self.pose_group,
            "goal_group": self.goal_group,
        }

        return observation, info

    def preprocess_action(self, action):
        if symmetry:
            if mapping != 0:
                action[:3] *= 1 /(self.action_scale_rot*100)
                action[3:] *= 1 /(self.action_scale_pos*100)

            else:
                pass

        else:
            if mapping != 0:
                pass
            else:
               pass 
                

        return action

    def increase_pose(self, delta):

        pose_group = torch.tensor(self.pose_group).unsqueeze(0) 
        goal_group = torch.tensor(self.goal_group).unsqueeze(0)

        pose_map = torch.tensor(self.pose_map).unsqueeze(0) 
        goal_map = torch.tensor(self.goal_map).unsqueeze(0)

        if symmetry:
            if mapping != 0:
                diff = self.log(self.diff_operator(goal_group, pose_group))
                res_diff = self.exp(diff + delta)

                self.pose_group = self.mul_operator(goal_group, res_diff)
                self.pose_map = self.log(pose_group)
                self.pose = self.convert_to_Lab(self.pose_group).squeeze(0)             
                

            else:
                diff = self.diff_operator(goal_group, pose_group)
                res_diff = self.mul_operator(pose_group, diff)
                
                self.pose_group = self.mul_operator(goal_group, res_diff)
                self.pose_map = self.log(pose_group)

        else:
            if mapping != 0:
                self.pose_map = pose_map + delta
                self.pose_group = self.log(self.pose_map)
                
            else:
                self.pose_group = self.mul_operator(pose_group, delta)
                self.pose_map = self.log(pose_group)


        self.pose_group = self.pose_group.squeeze(0).cpu().numpy()
        self.pose_map = self.pose_map.squeeze(0).cpu().numpy()

    def step(self, action):
        """
        Apply one pose increment.
        """

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        self.action = action
        

        # --------------------------------------------------
        # Convert normalized action [-1, 1] into increment
        # --------------------------------------------------
        delta_pose = self.preprocess_action(action)

        # --------------------------------------------------
        # Update pose
        # --------------------------------------------------
        self.increase_pose(delta_pose)

        self.pose_map_history.append(self.pose_map)

        # TODO: normalise
        # --------------------------------------------------
        # Keep pose inside workspace / Normalise
        # --------------------------------------------------

        self.visualize_pose_evolution(show=False, save_path="./evo")
        

        self.step_count += 1

        # --------------------------------------------------
        # Reward
        # --------------------------------------------------

        reward = self.compute_reward()

        # --------------------------------------------------
        # Termination
        # --------------------------------------------------

        terminated = self.check_goal()

        out_bounds = np.any(self.pose[:3].cpu().numpy() <= self.pos_low) and np.any(self.pose[:3].cpu().numpy() >= self.pos_high)
        time_out = self.step_count >= self.end_episode

        truncated = self.step_count >= self.max_steps and out_bounds and time_out

        # Optional terminal bonus
        if terminated:
            reward += 10.0

        # --------------------------------------------------
        # Information
        # --------------------------------------------------

        position_distance, rotation_distance = (
            self.compute_distance()
        )

        info = {
            "pose": self.pose,
            "goal": self.goal,
            "position_distance": position_distance,
            "rotation_distance": rotation_distance,
        }

        return (
            self.get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def visualize_pose_evolution(self, show=True, save_path=None):
        """
        Visualize the evolution of the pose in Lie algebra coordinates.

        pose_map convention:
            pose_map[:3] -> rotational Lie algebra coordinates
            pose_map[3:] -> translational Lie algebra coordinates

        The trajectory stored in self.pose_map_history is plotted together
        with the goal pose.
        """

        if len(self.pose_map_history) == 0:
            print("No pose history available.")
            return

        trajectory = np.asarray(self.pose_map_history)

        # Goal in Lie algebra
        goal = np.asarray(self.goal_map)

        # ------------------------------------------------------------
        # Create figure
        # ------------------------------------------------------------

        fig = plt.figure(figsize=(15, 10))

        # ============================================================
        # 1. 3D POSITION TRAJECTORY
        # ============================================================

        ax1 = fig.add_subplot(221, projection="3d")

        x = trajectory[:, 3]
        y = trajectory[:, 4]
        z = trajectory[:, 5]

        gx, gy, gz = goal[3], goal[4], goal[5]

        ax1.plot(
            x,
            y,
            z,
            linewidth=2,
            label="Pose trajectory",
        )

        ax1.scatter(
            x[0],
            y[0],
            z[0],
            s=80,
            marker="o",
            label="Start",
        )

        ax1.scatter(
            gx,
            gy,
            gz,
            s=120,
            marker="*",
            label="Goal",
        )

        ax1.set_xlabel(r"$\xi_4$")
        ax1.set_ylabel(r"$\xi_5$")
        ax1.set_zlabel(r"$\xi_6$")

        ax1.set_title("Translation trajectory in Lie algebra")

        ax1.legend()

        # ============================================================
        # 2. ROTATION LIE ALGEBRA
        # ============================================================

        ax2 = fig.add_subplot(222)

        steps = np.arange(len(trajectory))

        ax2.plot(
            steps,
            trajectory[:, 0],
            label=r"$\omega_x$",
        )

        ax2.plot(
            steps,
            trajectory[:, 1],
            label=r"$\omega_y$",
        )

        ax2.plot(
            steps,
            trajectory[:, 2],
            label=r"$\omega_z$",
        )

        ax2.axhline(
            goal[0],
            linestyle="--",
            label=r"Goal $\omega_x$",
        )

        ax2.axhline(
            goal[1],
            linestyle="--",
            label=r"Goal $\omega_y$",
        )

        ax2.axhline(
            goal[2],
            linestyle="--",
            label=r"Goal $\omega_z$",
        )

        ax2.set_xlabel("Step")
        ax2.set_ylabel("Lie algebra coordinate")

        ax2.set_title("Rotation evolution")

        ax2.grid(True)
        ax2.legend()

        # ============================================================
        # 3. TRANSLATION LIE ALGEBRA
        # ============================================================

        ax3 = fig.add_subplot(223)

        ax3.plot(
            steps,
            trajectory[:, 3],
            label=r"$v_x$",
        )

        ax3.plot(
            steps,
            trajectory[:, 4],
            label=r"$v_y$",
        )

        ax3.plot(
            steps,
            trajectory[:, 5],
            label=r"$v_z$",
        )

        ax3.axhline(
            goal[3],
            linestyle="--",
            label=r"Goal $v_x$",
        )

        ax3.axhline(
            goal[4],
            linestyle="--",
            label=r"Goal $v_y$",
        )

        ax3.axhline(
            goal[5],
            linestyle="--",
            label=r"Goal $v_z$",
        )

        ax3.set_xlabel("Step")
        ax3.set_ylabel("Lie algebra coordinate")

        ax3.set_title("Translation evolution")

        ax3.grid(True)
        ax3.legend()

        # ============================================================
        # 4. DISTANCE TO GOAL IN LIE ALGEBRA
        # ============================================================

        ax4 = fig.add_subplot(224)

        diff = trajectory - goal

        rotation_error = np.linalg.norm(
            diff[:, :3],
            axis=1,
        )

        translation_error = np.linalg.norm(
            diff[:, 3:],
            axis=1,
        )

        total_error = np.linalg.norm(
            diff,
            axis=1,
        )

        ax4.plot(
            steps,
            rotation_error,
            label="Rotation error",
        )

        ax4.plot(
            steps,
            translation_error,
            label="Translation error",
        )

        ax4.plot(
            steps,
            total_error,
            linewidth=2,
            label="Total Lie error",
        )

        ax4.set_xlabel("Step")
        ax4.set_ylabel("Distance")

        ax4.set_title("Distance to goal in Lie algebra")

        ax4.grid(True)
        ax4.legend()

        # ------------------------------------------------------------
        # Final formatting
        # ------------------------------------------------------------

        fig.suptitle(
            "Pose evolution in Lie algebra",
            fontsize=16,
        )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(
                save_path,
                dpi=200,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        return fig