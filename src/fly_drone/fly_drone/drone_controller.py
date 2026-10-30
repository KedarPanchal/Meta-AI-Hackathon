import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

import random

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class DroneController(Node):
    def __init__(self):
        super().__init__("drone_controller")
        
        # Create single-use publisher for enabling the drone
        enable_publisher = self.create_publisher(Bool, "enable", 1)
        # Start the drone
        enable_publisher.publish(Bool(data=True))

        self._publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self._motion_subscription = self.create_subscription(
            Twist,
            "drone_motion",
            self.issue_command,
            10
        )
    
    def issue_command(self, msg: Twist):
        # Forward message to Gazebo
        self._publisher.publish(msg)
        self.get_logger().info(
            f"Published linear vector: \n"
            f"x: {msg.linear.x:.2f}, "
            f"y: {msg.linear.y:.2f}, "
            f"z: {msg.linear.z:.2f}, \n"
            f"Published angular vector: \n"
            f"angular x: {msg.angular.x:.2f}, "
            f"angular y: {msg.angular.y:.2f}, "
            f"angular z: {msg.angular.z:.2f}"
        )


def main(args=None):
    try:
        with rclpy.init(args=args):
            drone_controller = DroneController()
            rclpy.spin(drone_controller)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
