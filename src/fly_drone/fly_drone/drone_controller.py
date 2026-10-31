import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

import random

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class DroneController(Node):
    def __init__(self):
        super().__init__("drone_controller")
        
        self._publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self._motion_subscription = self.create_subscription(
            Twist,
            "drone_motion",
            self.issue_command,
            10
        )

        # Create single-use publisher for enabling the drone
        self._enable_publisher = self.create_publisher(Bool, "enable", 1)
        self._enabled = False

        # Create consistent, time-based control loop
        self._control_timer = self.create_timer(0.05, self.periodic_control)
        self._previous_twist = Twist()
    
    def issue_command(self, msg: Twist):
        # Forward message to Gazebo
        self._previous_twist = msg

    def periodic_control(self):
        if (
            self._publisher.get_subscription_count() == 0
            or self._enable_publisher.get_subscription_count() == 0
        ):
            self.get_logger().info("Waiting for Gazebo subscribers to connect...")
            return
        
        if not self._enabled:
            self._enable_publisher.publish(Bool(data=True))
            self._enabled = True
            self.get_logger().info("Drone enabled")

        self._publisher.publish(self._previous_twist)
        self.get_logger().info(
            f"Published linear vector: \n"
            f"x: {self._previous_twist.linear.x:.2f}, "
            f"y: {self._previous_twist.linear.y:.2f}, "
            f"z: {self._previous_twist.linear.z:.2f}, \n"
            f"Published angular vector: \n"
            f"angular x: {self._previous_twist.angular.x:.2f}, "
            f"angular y: {self._previous_twist.angular.y:.2f}, "
            f"angular z: {self._previous_twist.angular.z:.2f}"
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
