import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.timer = self.create_timer(0.5, self.timer_callback)

        self.i = 0

        # Create the trajectory message
        self.msg = JointTrajectory()

        self.msg.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        # Create a trajectory point
        self.point = JointTrajectoryPoint()

        # Set joint velocities (rad/s)
        self.point.velocities = [0.0] * 6
        self.point.positions = [0.0] * 6

        # Set the trajectory execution time
        self.point.time_from_start = Duration(
            sec=0,
            nanosec=100_000_000
        )

        # Add the point to the trajectory
        self.msg.points = [self.point]

    def timer_callback(self):

        self.publisher_.publish(self.msg)

        self.get_logger().info(
            'Publishing: "%s"' % self.msg
        )

        self.i += 1


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