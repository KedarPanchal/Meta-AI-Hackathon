from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
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

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch_path),
            launch_arguments={
                "gz_args": PathJoinSubstitution([project_root, "worlds", "forest_search.world"]),
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
        )
    ])
