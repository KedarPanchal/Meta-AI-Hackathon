"""
Gazebo plus the fly brain flying the X3: ``ros2 launch fly_bringup brain_launch.py``.

Everything ``sim_launch.py`` starts (the world, the bridges, the drone_controller, the chase
camera and the video), and on top of it:

* ``brain_pilot`` (``fly_drone``): the connectome brain as the pilot. It subscribes to the
  odometry, the forward camera and its camera_info, and the lidar; publishes velocity on
  ``/drone/drone_motion`` (forwarded to the vehicle by ``drone_controller``); takes goals on
  ``/fly_brain/goal``. Output is disarmed until a goal arrives (``arm_on_goal``) or
  ``enable_output:=true``.
* ``threat_launcher`` (optional, ``threats:=true``): throws spheres at the drone so the escape
  reflex has something to escape from. ``threat_period_s:=0`` launches only on request
  (``/threats/launch``). Comes with a second ``ros_gz_bridge`` that exposes Gazebo's
  ``create``, ``remove`` and ``set_pose`` services to ROS.

Arguments::

    backend:=mock|flycns      BRAIN_BACKEND (default mock; flycns needs the FLYCNS clone)
    flycns_repo:=PATH         FLYDRONE_REPO (default ../FLYCNS next to this repository)
    flycns_data:=testdata     FLYDRONE_DATA (testdata: the 912-neuron toy graph)
    enable_output:=false      arm the output at start, rather than on the first goal
    arm_on_goal:=true
    threats:=false            start the threat launcher
    threat_period_s:=20       seconds between launches (0: only on /threats/launch)
    threat_delay_s:=15        seconds before the first launch
    use_sim_time:=true        the brain pilot runs on Gazebo's clock (bridged on /clock)
    headless:=false           no GUI (sensors still render); passed to sim_launch.py
    gazebo_simulation, camera_tracking, video_recording, video_dir: passed to sim_launch.py

The mission layer (the MCP server in ``mcp-server/``) is not started here: an MCP client
starts it, and it reaches the brain over the topics above.
"""
from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    project_root = os.getenv('PIXI_PROJECT_ROOT', os.getcwd())
    flycns_default = os.path.normpath(os.path.join(project_root, '..', 'FLYCNS'))

    args = [
        DeclareLaunchArgument('backend', default_value='mock',
                              description='BRAIN_BACKEND: mock or flycns'),
        DeclareLaunchArgument('flycns_repo', default_value=flycns_default,
                              description='FLYDRONE_REPO: the FLYCNS clone'),
        DeclareLaunchArgument('flycns_data', default_value='testdata',
                              description='FLYDRONE_DATA: graph directory inside the clone'),
        DeclareLaunchArgument('enable_output', default_value='false',
                              description='Arm the brain output at start'),
        DeclareLaunchArgument('arm_on_goal', default_value='true',
                              description='A goal on /fly_brain/goal arms the output'),
        DeclareLaunchArgument('threats', default_value='false',
                              description='Start the threat launcher'),
        DeclareLaunchArgument('threat_period_s', default_value='20.0',
                              description='Seconds between launches; 0 = only on request'),
        DeclareLaunchArgument('threat_delay_s', default_value='15.0',
                              description='Seconds before the first launch'),
        DeclareLaunchArgument('threat_distance_m', default_value='10.0'),
        DeclareLaunchArgument('threat_speed_mps', default_value='5.0'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Run the brain pilot on the simulation clock'),
        # pass-through to sim_launch.py
        DeclareLaunchArgument('gazebo_simulation', default_value='forest_search'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('camera_tracking', default_value='true'),
        DeclareLaunchArgument('camera_tracking_target', default_value='x3_search_uav'),
        DeclareLaunchArgument('video_recording', default_value='true'),
        DeclareLaunchArgument('video_dir', default_value=os.path.join(project_root, 'dist')),
    ]
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('fly_bringup'), 'launch', 'sim_launch.py'])
        ),
        launch_arguments={
            k: LaunchConfiguration(k)
            for k in ('gazebo_simulation', 'headless', 'camera_tracking',
                      'camera_tracking_target', 'video_recording', 'video_dir')
        }.items(),
    )
    brain = Node(
        package='fly_drone',
        executable='brain_pilot',
        name='fly_brain_pilot',
        output='screen',
        parameters=[{
            'enable_output': ParameterValue(LaunchConfiguration('enable_output'), value_type=bool),
            'arm_on_goal': ParameterValue(LaunchConfiguration('arm_on_goal'), value_type=bool),
            'use_sim_time': use_sim_time,
        }],
        additional_env={
            'BRAIN_BACKEND': LaunchConfiguration('backend'),
            'FLYDRONE_REPO': LaunchConfiguration('flycns_repo'),
            'FLYDRONE_DATA': LaunchConfiguration('flycns_data'),
        },
    )
    period = ParameterValue(LaunchConfiguration('threat_period_s'), value_type=float)
    world = LaunchConfiguration('gazebo_simulation')
    # Gazebo's entity services over ROS, for the threat launcher: spawn, remove, set_pose.
    services = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_entity_services',
        output='screen',
        condition=IfCondition(LaunchConfiguration('threats')),
        arguments=[
            ['/world/', world, '/create@ros_gz_interfaces/srv/SpawnEntity'],
            ['/world/', world, '/remove@ros_gz_interfaces/srv/DeleteEntity'],
            ['/world/', world, '/set_pose@ros_gz_interfaces/srv/SetEntityPose'],
        ],
    )
    threats = Node(
        package='fly_drone',
        executable='threat_launcher',
        name='threat_launcher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('threats')),
        parameters=[{
            'world': world,
            'use_sim_time': use_sim_time,
            'mode': 'periodic',
            'period_s': period,
            'delay_s': ParameterValue(LaunchConfiguration('threat_delay_s'), value_type=float),
            'distance_m': ParameterValue(LaunchConfiguration('threat_distance_m'), value_type=float),
            'speed_mps': ParameterValue(LaunchConfiguration('threat_speed_mps'), value_type=float),
        }],
    )
    return LaunchDescription(args + [sim, brain, services, threats])
