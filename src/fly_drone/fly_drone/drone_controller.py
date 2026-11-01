import rclpy
import random

from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, PointCloud2


class DroneController(Node):
    def __init__(self):
        super().__init__("drone_controller")
        
        # Subscribe to motion controls to forward to Gazebo
        self._publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self._motion_subscription = self.create_subscription(
            Twist,
            "drone_motion",
            self.issue_command,
            10
        )

        # Subscribe to sensor information and synchronize the messages
        self._camera_image_subscription = Subscriber(self, Image, "camera_image")
        self._lidar_subscription = Subscriber(self, PointCloud2, "lidar_pointcloud")
        self._raw_synchronizer = ApproximateTimeSynchronizer(
            [self._camera_image_subscription, self._lidar_subscription],
            queue_size=10,
            slop=0.1,
        )
        self._raw_synchronizer.registerCallback(self.raw_sensor_control)

        # Create single-use publisher for enabling the drone
        self._enable_publisher = self.create_publisher(Bool, "enable", 1)
        self._enabled = False

        # Create consistent, time-based control loop
        self._control_timer = self.create_timer(0.05, self.periodic_control)
        self._previous_twist = Twist()

    def raw_sensor_control(self, camera_image: Image, lidar_pointcloud: PointCloud2):
        self.get_logger().info(
            f"Received camera image with "
            f"Dimensions: {camera_image.width}x{camera_image.height}, "
            f"Encoding: {camera_image.encoding}, "
            f"Data length: {len(camera_image.data)} bytes\n"
            f"Received LIDAR point cloud with width: {lidar_pointcloud.width}, "
            f"height: {lidar_pointcloud.height}, "
            f"Point step: {lidar_pointcloud.point_step}, "
            f"Row step: {lidar_pointcloud.row_step}, "
            f"Data length: {len(lidar_pointcloud.data)} bytes"
        )
    
    def issue_command(self, msg: Twist):
        # Set message to forward to Gazebo in the next control loop iteration
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
            f"pitch: {self._previous_twist.angular.x:.2f}, "
            f"yaw: {self._previous_twist.angular.y:.2f}, "
            f"roll: {self._previous_twist.angular.z:.2f}"
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
