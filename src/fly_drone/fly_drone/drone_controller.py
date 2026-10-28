import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

# TODO: Convert this to an actual custom message. For now, use a String placeholder
from std_msgs.msg import String


class DroneController(Node):
    def __init__(self):
        super().__init__("drone_controller")
        self._publisher = self.create_publisher(String, "drone_commands", 10)
        self.timer = self.create_timer(1.0, self.issue_command)
    
    def issue_command(self):
        msg = String()
        msg.data = "Take off"
        self._publisher.publish(msg)
        self.get_logger().info(f"Published command: {msg.data}")


def main(args=None):
    try:
        with rclpy.init(args=args):
            drone_controller = DroneController()
            rclpy.spin(drone_controller)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
