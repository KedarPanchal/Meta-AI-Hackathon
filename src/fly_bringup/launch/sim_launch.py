from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

import os


def generate_launch_description():
    project_root = os.getenv("PIXI_PROJECT_ROOT", os.getcwd())
    bridge_config = PathJoinSubstitution([
        FindPackageShare("fly_bringup"),
        "config",
        "bridge.yaml"
    ])
    gazebo_launch_path = PathJoinSubstitution([
        get_package_share_directory("ros_gz_sim"),
        "launch",
        "gz_sim.launch.py"
    ])
    gazebo_simulation = DeclareLaunchArgument(
        "gazebo_simulation",
        default_value="forest_search",
        description="Gazebo simulation world to launch"
    )
    camera_tracking = DeclareLaunchArgument(
        "camera_tracking",
        default_value="true",
        description="Follow the drone with the Gazebo GUI camera"
    )
    camera_tracking_target = DeclareLaunchArgument(
        "camera_tracking_target",
        default_value="x3_search_uav",
        description="Name of the Gazebo model the GUI camera follows"
    )

    return LaunchDescription([
        gazebo_simulation,
        camera_tracking,
        camera_tracking_target,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch_path),
            launch_arguments={
                "gz_args": ["-r ", PathJoinSubstitution([project_root, "worlds", [LaunchConfiguration("gazebo_simulation"), ".world"]])],
                "on_exit_shutdown": "True"
            }.items()
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            parameters=[{"config_file": bridge_config}],
            output="screen",
        ),
        Node(
            package="fly_drone",
            namespace="drone",
            executable="drone_controller",
            arguments=["--ros-args", "--log-level", "info"],
            output="screen",
        ),
        Node(
            package="fly_bringup",
            executable="follow_camera",
            parameters=[{"target": LaunchConfiguration("camera_tracking_target")}],
            condition=IfCondition(LaunchConfiguration("camera_tracking")),
            output="screen",
        )
    ])
