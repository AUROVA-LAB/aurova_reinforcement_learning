import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from .mpc_controller import *
import time


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        # Joint names in the order expected by the trajectory controller
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        # Store the latest joint positions
        self.current_positions = None

        # Subscribe to joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Publisher for joint trajectories
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.dt = 0.01

        # Timer
        self.timer = self.create_timer(self.dt, self.timer_callback)

        # Create the trajectory message
        self.msg = JointTrajectory()
        self.msg.joint_names = self.joint_names

        # Create a trajectory point
        self.point = JointTrajectoryPoint()

        self.point.positions = [0.0] * 6
        self.point.velocities = [0.0] * 6

        # Trajectory execution time: 0.1 seconds
        self.point.time_from_start = Duration(
            sec=0,
            nanosec=10_000_000
        )

        self.controller = drop_NMPC_setup(
            ini=[0.0] * 6,
            ref=[-0.49,  -0.13, 0.4088,  
                 -0.0190,      -0.3845,     -0.9191,      0.0843],
            dt=self.dt,
            horizon=30,
        )


        self.max_steps=1000
        self.control_dt=self.dt

        self.opti = self.controller["opti"]
        self.X = self.controller["X"]
        self.U = self.controller["U"]
    
        self.q_parameter = self.controller["q_measured"]
        self.u_previous_parameter = self.controller["u_previous"]
    
        self.T_ref = self.controller["T_ref"]
    
        self.dt = (
            self.controller["dt"]
            if self.control_dt is None
            else self.control_dt
        )
    
        if self.dt <= 0:
            raise ValueError("control_dt must be positive.")
    
        self.horizon = self.controller["horizon"]
    
        # Previous applied velocity command.
        self.u_previous = np.zeros(6)
    
        # Initial solver guesses.
        self.X_guess = np.tile(
            np.asarray([0,0,0,0,0,0], dtype=float).reshape(6, 1),
            (1, self.horizon+1),
        )
    
        self.U_guess = np.zeros((6, self.horizon))

        self.msg.points = [self.point]

    def joint_state_callback(self, msg: JointState):
        """Update the current positions using the joint names."""

        # Map each joint name to its current position
        joint_position_map = dict(zip(msg.name, msg.position))

        # Ensure all required joints are available
        if not all(name in joint_position_map for name in self.joint_names):
            self.get_logger().warning(
                "Waiting for all required joint states."
            )
            return

        # Store positions in the controller's expected order
        self.current_positions = [
            joint_position_map[name]
            for name in self.joint_names
        ]


    def mpc_step(self,):

        # ------------------------------------------
        # 1. Read the latest robot state
        # ------------------------------------------
        q_measured = np.asarray(
            self.current_positions,
            dtype=float,
        ).reshape(6)

        if not np.all(np.isfinite(q_measured)):
            raise ValueError(
                "Received invalid joint positions."
            )

        # ------------------------------------------
        # 2. Update NMPC parameters
        # ------------------------------------------
        self.opti.set_value(
            self.q_parameter,
            q_measured,
        )

        self.opti.set_value(
            self.u_previous_parameter,
            self.u_previous,
        )

        # ------------------------------------------
        # 3. Warm-start the optimization
        # ------------------------------------------
        
        self.opti.set_initial(self.X, self.X_guess)
        self.opti.set_initial(self.U, self.U_guess)

        # ------------------------------------------
        # 4. Solve NMPC
        # ------------------------------------------
        solve_start = time.perf_counter()

        solution = self.opti.solve()

        solve_time = time.perf_counter() - solve_start

        # ------------------------------------------
        # 5. Extract the first optimal command
        # ------------------------------------------
        U_opt = solution.value(self.U)
        X_opt = solution.value(self.X)

        u_cmd = np.asarray(
            U_opt[:, 0],
            dtype=float,
        ).reshape(6)

        self.u_previous = u_cmd.copy()

        pose_error = np.asarray(
            self.controller["pose_error"](q_measured).full()
        ).reshape(6)

        position_error = np.linalg.norm(
            pose_error[:3]
        )

        orientation_error = np.linalg.norm(
            pose_error[3:]
        )

        print(
                f"position error={position_error:.4f} m | "
                f"orientation error={orientation_error:.4f} rad | "
                f"solve={solve_time * 1000:.1f} ms"
            )
        print("---------------")

        return u_cmd



    def timer_callback(self):
        """Publish a trajectory using the latest measured joint positions."""

        # Do not publish until joint states have been received
        if self.current_positions is None:
            self.get_logger().warning(
                "Waiting for joint states..."
            )
            return

        cmd = self.mpc_step()

        # Update the trajectory point with the current joint positions
        self.msg.points[0].positions = (self.current_positions.copy() + cmd*self.dt).tolist()
        
        # self.msg.points[0].velocities = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        
        # Publish the trajectory
        self.publisher_.publish(self.msg)

        # self.get_logger().info(
        #     f"Publishing joint positions: {self.msg.points[0].positions}"
        # )


def main(args=None):

    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    try:
        rclpy.spin(minimal_publisher)
    finally:
        minimal_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()