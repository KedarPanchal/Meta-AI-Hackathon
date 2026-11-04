from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable
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
    hd_recording = DeclareLaunchArgument(
        "hd_recording",
        default_value="false",
        description="Also record the dedicated HD chase camera "
                    "(1920x1080 at 60 Hz, models/x3_uav/model.sdf) to "
                    "<world>_<timestamp>_hd.mp4 in video_dir, via Gazebo's "
                    "CameraVideoRecorder service. Works with or without the "
                    "GUI. The chase-sensor recording (video_recording) is "
                    "unaffected."
    )
    headless = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Server only, no GUI window; the cameras and lidar still render "
                    "off-screen (EGL). For tests and for machines without a display. "
                    "Disables camera_tracking, which only concerns the GUI camera."
    )
    is_headless = PythonExpression(["'", LaunchConfiguration("headless"), "'.lower() in ('true', '1')"])

    # Workaround for the HD recorder plugin, applied only when it is used:
    # gz's VideoEncoder libraries pull in the pixi libsystemd, which needs
    # a symbol (removexattrat) this host's libc does not provide, so the
    # CameraVideoRecorder plugin fails to load. Preloading the host
    # libsystemd lets it resolve instead. Scoped to hd_recording so plain
    # launches are untouched, and skipped where the host path is absent.
    preload_systemd = []
    if os.path.exists("/usr/lib64/libsystemd.so.0"):
        preload_systemd = [SetEnvironmentVariable(
            "LD_PRELOAD", "/usr/lib64/libsystemd.so.0",
            condition=IfCondition(LaunchConfiguration("hd_recording")))]

    return LaunchDescription([
        gazebo_simulation,
        camera_tracking,
        camera_tracking_target,
        video_recording,
        video_dir,
        hd_recording,
        headless,
        *preload_systemd,
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
        # record_video only finalises its .partial.mp4 on a graceful shutdown:
        # it drains the frame queue, closes ffmpeg so the footer is written,
        # then renames the file. The launch default escalates SIGINT -> SIGTERM
        # after 5 s and SIGKILL 5 s later, which can kill it mid-finalise on a
        # long recording (padding/encoding the tail takes seconds). Give this
        # one node a long grace so the Ctrl-C path has time to finish; a truly
        # hung recorder is still escalated, just later.
        Node(
            package="fly_bringup",
            executable="record_video",
            parameters=[{
                "output_dir": LaunchConfiguration("video_dir"),
                "world": LaunchConfiguration("gazebo_simulation"),
            }],
            condition=IfCondition(LaunchConfiguration("video_recording")),
            sigterm_timeout="30",
            sigkill_timeout="10",
            output="screen",
        ),
        # HD recording: record_hd_video starts Gazebo's CameraVideoRecorder
        # on the dedicated HD chase camera over its gz service once the
        # world is up and stops it on shutdown, so the file is saved with
        # no GUI interaction. Its stop call races Gazebo's own exit on
        # Ctrl-C, hence the same long escalation grace as record_video.
        Node(
            package="fly_bringup",
            executable="record_hd_video",
            parameters=[{
                "output_dir": LaunchConfiguration("video_dir"),
                "world": LaunchConfiguration("gazebo_simulation"),
            }],
            condition=IfCondition(LaunchConfiguration("hd_recording")),
            sigterm_timeout="30",
            sigkill_timeout="10",
            output="screen",
        )
    ])
