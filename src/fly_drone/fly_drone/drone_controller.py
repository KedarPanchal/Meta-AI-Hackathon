import rclpy
import struct
from collections import deque
from math import hypot

from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber

from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from fly_interfaces.msg import GeometricObject, RawSensorData, SensorFusion
from .detect import Detector


class DroneController(Node):
    """Forward flight commands and publish synchronized camera/LIDAR perception."""

    def __init__(self):
        """Initialize control publishers, sensor subscriptions, and processing state."""
        super().__init__("drone_controller")

        self._init_motion()
        self._init_odometry()
        self._init_sensors()
        self._enable_drone()
        
        # Create consistent, time-based control loop
        self._control_timer = self.create_timer(0.05, self.periodic_control)
        self._previous_twist = Twist()

    def _init_motion(self): 
        # Subscribe to motion controls to forward to Gazebo
        self._motion_publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self._motion_subscription = self.create_subscription(
            Twist,
            "drone_motion",
            self.issue_command,
            10
        )

    def _init_odometry(self):
        # Subscribe to odometry information and process it for Muse model
        self._odometry_publisher = self.create_publisher(Pose, "position", 10)
        self._odometry_subscription = self.create_subscription(
            Odometry,
            "odometry",
            self.process_odometry,
            10
        )

    def _init_sensors(self):
        # Subscribe to sensor information and synchronize the messages
        self._camera_image_subscription = Subscriber(self, Image, "camera_image")
        self._camera_info_subscription = Subscriber(self, CameraInfo, "camera_info")
        self._lidar_subscription = Subscriber(self, PointCloud2, "lidar_pointcloud")
        self._sensor_synchronizer = ApproximateTimeSynchronizer(
            [
                self._camera_image_subscription,
                self._camera_info_subscription,
                self._lidar_subscription,
            ],
            queue_size=10,
            slop=0.1,
        )
        self._sensor_synchronizer.registerCallback(self.sensor_control)

        # Publish raw sensor data
        self._raw_sensor_publisher = self.create_publisher(RawSensorData, "raw_sensor_data", 10)
        self._sensor_fusion_publisher = self.create_publisher(
            SensorFusion, "sensor_fusion", 10
        )
        self._camera_detector = Detector()
        self._camera_bridge = None

    def _enable_drone(self):
        # Create single-use publisher for enabling the drone
        self._enable_publisher = self.create_publisher(Bool, "enable", 1)
        self._enabled = False

    def process_odometry(self, odometry: Odometry):
        drone_pose = odometry.pose.pose
        self.get_logger().info(
            f"Received odometry with position: "
            f"x: {drone_pose.position.x:.2f}, "
            f"y: {drone_pose.position.y:.2f}, "
            f"z: {drone_pose.position.z:.2f}, "
            f"orientation: "
            f"x: {drone_pose.orientation.x:.2f}, "
            f"y: {drone_pose.orientation.y:.2f}, "
            f"z: {drone_pose.orientation.z:.2f}, "
            f"w: {drone_pose.orientation.w:.2f}"
        )
        self._odometry_publisher.publish(drone_pose)

    def sensor_control(
        self,
        camera_image: Image,
        camera_info: CameraInfo,
        lidar_pointcloud: PointCloud2,
    ):
        """Fuse synchronized camera and LIDAR data and publish raw and derived results."""
        objects_from_camera = self._objects_from_camera(camera_image, camera_info)
        objects_below = self._objects_below_drone(lidar_pointcloud)
        self.get_logger().info(
            f"Received camera image with "
            f"Dimensions: {camera_image.width}x{camera_image.height}, "
            f"Encoding: {camera_image.encoding}, "
            f"Data length: {len(camera_image.data)} bytes, "
            f"calibration frame: {camera_info.header.frame_id}, "
            f"focal length: ({camera_info.k[0]:.2f}, {camera_info.k[4]:.2f})\n"
            f"Received LIDAR point cloud with width: {lidar_pointcloud.width}, "
            f"height: {lidar_pointcloud.height}, "
            f"Point step: {lidar_pointcloud.point_step}, "
            f"Row step: {lidar_pointcloud.row_step}, "
            f"Data length: {len(lidar_pointcloud.data)} bytes"
        )
        self._raw_sensor_publisher.publish(RawSensorData(
            camera_raw=camera_image,
            lidar_raw=lidar_pointcloud
        ))
        self._sensor_fusion_publisher.publish(SensorFusion(
            objects_from_camera=objects_from_camera,
            objects_below=objects_below,
        ))

    # Camera detections are projected into normalized camera coordinates so their
    # geometry remains useful without pretending that image depth is known.
    def _objects_from_camera(
        self, image: Image, camera_info: CameraInfo
    ) -> list[GeometricObject]:
        """Convert detected image regions into calibrated unit-depth geometry."""
        try:
            from cv_bridge import CvBridge

            if self._camera_bridge is None:
                self._camera_bridge = CvBridge()
            frame = self._camera_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'cannot convert camera image: {exc}', once=True)
            return []

        focal_x, focal_y = camera_info.k[0], camera_info.k[4]
        if focal_x <= 0.0 or focal_y <= 0.0:
            return []

        objects = []
        for box in self._camera_detector.detect(frame):
            center_x = box.x + box.w / 2.0
            center_y = box.y + box.h / 2.0
            object_geometry = GeometricObject()
            object_geometry.center.x = (center_x - camera_info.k[2]) / focal_x
            object_geometry.center.y = (center_y - camera_info.k[5]) / focal_y
            object_geometry.center.z = 1.0
            object_geometry.dimensions.x = box.w / focal_x
            object_geometry.dimensions.y = box.h / focal_y
            object_geometry.dimensions.z = 0.0
            object_geometry.point_count = max(1, int(box.w * box.h))
            objects.append(object_geometry)
        return objects

    # LIDAR supplies metric depth directly, so clusters can be represented as
    # axis-aligned bounding boxes in the point-cloud coordinate frame.
    def _objects_below_drone(self, pointcloud: PointCloud2) -> list[GeometricObject]:
        """Cluster valid points below the drone into simple geometric objects."""
        points = self._pointcloud_points(pointcloud)
        if not points:
            return []

        remaining = set(range(len(points)))
        objects = []
        while remaining:
            seed = remaining.pop()
            cluster = [seed]
            queue = deque([seed])
            while queue:
                current = queue.popleft()
                cx, cy, _ = points[current]
                neighbours = [
                    index for index in remaining
                    if hypot(points[index][0] - cx, points[index][1] - cy) <= 0.5
                ]
                remaining.difference_update(neighbours)
                queue.extend(neighbours)
                cluster.extend(neighbours)

            cluster_points = [points[index] for index in cluster]
            xs, ys, zs = zip(*cluster_points)
            geometric_object = GeometricObject()
            geometric_object.center.x = (min(xs) + max(xs)) / 2.0
            geometric_object.center.y = (min(ys) + max(ys)) / 2.0
            geometric_object.center.z = (min(zs) + max(zs)) / 2.0
            geometric_object.dimensions.x = max(xs) - min(xs)
            geometric_object.dimensions.y = max(ys) - min(ys)
            geometric_object.dimensions.z = max(zs) - min(zs)
            geometric_object.point_count = len(cluster_points)
            geometric_object.min_z = min(zs)
            geometric_object.max_z = max(zs)
            objects.append(geometric_object)

        return objects

    @staticmethod
    def _pointcloud_points(pointcloud: PointCloud2) -> list[tuple[float, float, float]]:
        """Read finite XYZ points and retain only points below the drone."""
        fields = {field.name: field for field in pointcloud.fields}
        if not all(name in fields for name in ("x", "y", "z")):
            return []

        offsets = tuple(fields[name].offset for name in ("x", "y", "z"))
        endian = ">" if pointcloud.is_bigendian else "<"
        points = []
        for row in range(pointcloud.height):
            row_start = row * pointcloud.row_step
            for column in range(pointcloud.width):
                point_start = row_start + column * pointcloud.point_step
                try:
                    point = tuple(
                        struct.unpack_from(f"{endian}f", pointcloud.data, point_start + offset)[0]
                        for offset in offsets
                    )
                except struct.error:
                    continue
                if all(value == value and abs(value) != float("inf") for value in point):
                    if point[2] <= 0.0:
                        points.append(point)
        return points
    
    def issue_command(self, msg: Twist):
        """Store the latest requested motion command for the control timer."""
        # Set message to forward to Gazebo in the next control loop iteration
        self._previous_twist = msg

    def periodic_control(self):
        """Enable the drone and repeatedly forward the latest motion command."""
        if (
            self._motion_publisher.get_subscription_count() == 0
            or self._enable_publisher.get_subscription_count() == 0
        ):
            self.get_logger().info("Waiting for Gazebo subscribers to connect...")
            return
        
        if not self._enabled:
            self._enable_publisher.publish(Bool(data=True))
            self._enabled = True
            self.get_logger().info("Drone enabled")

        self._motion_publisher.publish(self._previous_twist)
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
    """Start the drone controller node and keep it active until shutdown."""
    try:
        with rclpy.init(args=args):
            drone_controller = DroneController()
            rclpy.spin(drone_controller)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
