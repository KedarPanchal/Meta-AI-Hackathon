from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, PythonExpression
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
    chase_bridge_config = PathJoinSubstitution([
        FindPackageShare("fly_bringup"),
        "config",
        "chase_bridge.yaml"
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
    video_recording = DeclareLaunchArgument(
        "video_recording",
        default_value="true",
        description="Record the Gazebo chase-camera view and save it to video_dir on close"
    )
    video_dir = DeclareLaunchArgument(
        "video_dir",
        default_value=os.path.join(project_root, "dist"),
        description="Directory the recorded simulation video is written to"
    )
    headless = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Server only, no GUI window; the cameras and lidar still render "
                    "off-screen (EGL). For tests and for machines without a display. "
                    "Disables camera_tracking, which only concerns the GUI camera."
    )
    is_headless = PythonExpression(["'", LaunchConfiguration("headless"), "'.lower() in ('true', '1')"])

    return LaunchDescription([
        gazebo_simulation,
        camera_tracking,
        camera_tracking_target,
        video_recording,
        video_dir,
        headless,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch_path),
            launch_arguments={
                "gz_args": [
                    "-r ",
                    PythonExpression(["'-s --headless-rendering ' if ", is_headless, " else ''"]),
                    PathJoinSubstitution([project_root, "worlds", [LaunchConfiguration("gazebo_simulation"), ".world"]]),
                ],
                # gz_sim.launch.py defaults on_exit_shutdown to false. Without this,
                # closing Gazebo leaves the launch (and its bridges) running.
                "on_exit_shutdown": "True"
            }.items()
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            parameters=[{"config_file": bridge_config}],
            output="screen",
        ),
        # The chase image gets its own bridge. Sharing the main bridge with the
        # front camera and lidar starves it: the sensor renders, but almost no
        # images reach ROS, so the recording is only a frame or two.
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            parameters=[{"config_file": chase_bridge_config}],
            condition=IfCondition(LaunchConfiguration("video_recording")),
            output="screen",
        ),
        Node(
            package="fly_drone",
            namespace="drone",
            executable="drone_controller",
            output="screen",
        ),
        Node(
            package="fly_bringup",
            executable="follow_camera",
            parameters=[{"target": LaunchConfiguration("camera_tracking_target")}],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration("camera_tracking"), "'.lower() in ('true', '1') and not ", is_headless,
            ])),
            output="screen",
        ),
        Node(
            package="fly_bringup",
            executable="record_video",
            parameters=[{
                "output_dir": LaunchConfiguration("video_dir"),
                "world": LaunchConfiguration("gazebo_simulation"),
            }],
            condition=IfCondition(LaunchConfiguration("video_recording")),
            output="screen",
        )
    ])
