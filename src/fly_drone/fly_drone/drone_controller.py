import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

import random

from geometry_msgs.msg import Twist


class DroneController(Node):
    def __init__(self):
        super().__init__("drone_controller")
        self._publisher = self.create_publisher(Twist, "drone_commands", 10)
        self.timer = self.create_timer(1.0, self.issue_command)
    
    def issue_command(self):
        msg = Twist()
        msg.linear.x = random.uniform(-1.0, 1.0)
        msg.linear.y = random.uniform(-1.0, 1.0)
        msg.linear.z = random.uniform(-1.0, 1.0)
        msg.angular.x = random.uniform(-1.0, 1.0)
        msg.angular.y = random.uniform(-1.0, 1.0)
        msg.angular.z = random.uniform(-1.0, 1.0)

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
