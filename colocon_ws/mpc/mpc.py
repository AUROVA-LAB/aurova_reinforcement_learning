import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


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

        # Timer
        self.timer = self.create_timer(0.5, self.timer_callback)

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
            nanosec=100_000_000
        )

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

    def timer_callback(self):
        """Publish a trajectory using the latest measured joint positions."""

        # Do not publish until joint states have been received
        if self.current_positions is None:
            self.get_logger().warning(
                "Waiting for joint states..."
            )
            return

        # Update the trajectory point with the current joint positions
        self.msg.points[0].positions = self.current_positions.copy()
        self.msg.points[0].velocities = [0.7, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Publish the trajectory
        self.publisher_.publish(self.msg)

        self.get_logger().info(
            f"Publishing joint positions: {self.msg.points[0].positions}"
        )


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