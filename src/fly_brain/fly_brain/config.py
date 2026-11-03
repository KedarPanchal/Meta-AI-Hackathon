"""
Environment-driven settings.

Never name a setting ``FLYCNS_*``: FLYCNS's own ``announce_overrides()`` treats any such
variable as a declared perturbation of the model and stamps it into every report. Ours are
``FLYDRONE_*``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _f(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _i(name: str, default: int) -> int:
    return int(_env(name, str(default)))


@dataclass
class Settings:
    """Everything tunable from the environment, read once per process."""

    # --- brain ---------------------------------------------------------------
    backend: str = field(default_factory=lambda: _env('BRAIN_BACKEND', 'mock'))
    repo: Path = field(default_factory=lambda: Path(_env('FLYDRONE_REPO', '')))
    data: str = field(default_factory=lambda: _env('FLYDRONE_DATA', 'testdata'))
    circuit_path: str = field(default_factory=lambda: _env('FLYDRONE_CIRCUIT', ''))
    brain_tick_ms: int = field(default_factory=lambda: _i('BRAIN_TICK_MS', 50))

    # --- control loop --------------------------------------------------------
    loop_hz: float = field(default_factory=lambda: _f('LOOP_HZ', 20.0))

    # --- sensors -------------------------------------------------------------
    # Forward camera HFOV, degrees. Only a fallback: the ROS node reads the real value from
    # the camera_info topic (the X3's camera_front is 60 degrees, fx 554.3 at 640 px).
    camera_hfov_deg: float = field(default_factory=lambda: _f('CAMERA_HFOV_DEG', 60.0))
    # The 3-D lidar on the airframe (32 planes over +-30 degrees) feeds the loom tracker too,
    # as range clusters with elevation. FLYDRONE_LIDAR=0 flies on the camera alone.
    lidar_enabled: bool = field(
        default_factory=lambda: _env('FLYDRONE_LIDAR', '1').strip().lower() in ('1', 'true', 'yes', 'on')
    )
    lidar_max_range_m: float = field(default_factory=lambda: _f('FLYDRONE_LIDAR_RANGE_M', 20.0))
    # Returns more than this far below the airframe are the ground seen through a tilted scan
    # plane, not objects at the drone's level.
    lidar_max_below_m: float = field(default_factory=lambda: _f('FLYDRONE_LIDAR_MAX_BELOW_M', 2.0))
    # ...and more than this far above it are the canopy, or a bird that is not coming down.
    lidar_max_above_m: float = field(default_factory=lambda: _f('FLYDRONE_LIDAR_MAX_ABOVE_M', 4.0))

    # --- safety envelope -----------------------------------------------------
    max_speed: float = field(default_factory=lambda: _f('MAX_SPEED', 5.0))
    max_climb: float = field(default_factory=lambda: _f('MAX_CLIMB', 2.0))
    max_yaw: float = field(default_factory=lambda: _f('MAX_YAW', 2.0))
    min_altitude_m: float = field(default_factory=lambda: _f('MIN_ALTITUDE_M', 1.0))

    # --- ROS 2 interface -----------------------------------------------------
    # Default target is plain Gazebo through ros_gz_bridge, body-frame Twists. The names are
    # the ones src/fly_bringup/config/bridge.yaml creates under /drone: the brain publishes
    # velocity on /drone/drone_motion, which the drone_controller node forwards to
    # /drone/cmd_vel (the bridged topic) together with the enable flag.
    ros_bridge: str = field(default_factory=lambda: _env('ROS_BRIDGE', 'twist'))
    topic_cmd_vel: str = field(
        default_factory=lambda: _env('ROS_TOPIC_CMD_VEL', '/drone/drone_motion')
    )
    topic_odom: str = field(default_factory=lambda: _env('ROS_TOPIC_ODOM', '/drone/odometry'))
    topic_image: str = field(
        default_factory=lambda: _env('ROS_TOPIC_IMAGE', '/drone/camera_image')
    )
    topic_camera_info: str = field(
        default_factory=lambda: _env('ROS_TOPIC_CAMERA_INFO', '/drone/camera_info')
    )
    topic_lidar: str = field(
        default_factory=lambda: _env('ROS_TOPIC_LIDAR', '/drone/lidar3d_pointcloud')
    )
    topic_audio: str = field(default_factory=lambda: _env('ROS_TOPIC_AUDIO', '/drone/audio'))
    topic_image_down: str = field(
        default_factory=lambda: _env('ROS_TOPIC_IMAGE_DOWN', '/drone/camera_down_image')
    )
    topic_enable: str = field(default_factory=lambda: _env('ROS_TOPIC_ENABLE', '/drone/enable'))
    topic_telemetry: str = field(
        default_factory=lambda: _env('ROS_TOPIC_TELEMETRY', '/fly_brain/telemetry')
    )
    frame_id: str = field(default_factory=lambda: _env('ROS_FRAME_ID', 'base_link'))

    # --- the mission layer's two topics ---------------------------------------
    # Both carry JSON (std_msgs/String): a fly_brain.schemas.Goal down, a GoalStatus up. The
    # mission layer only ever says WHERE to go; velocity is the brain's business.
    topic_goal: str = field(default_factory=lambda: _env('ROS_TOPIC_GOAL', '/fly_brain/goal'))
    topic_goal_status: str = field(
        default_factory=lambda: _env('ROS_TOPIC_GOAL_STATUS', '/fly_brain/goal_status')
    )

    # --- goal navigation -----------------------------------------------------
    nav_speed_mps: float = field(default_factory=lambda: _f('NAV_SPEED_MPS', 2.5))
    nav_arrive_radius_m: float = field(default_factory=lambda: _f('NAV_ARRIVE_RADIUS_M', 1.5))

    def flycns_data_dir(self) -> Path:
        """Absolute path to the graph directory (``data`` or ``testdata``) inside the clone."""
        d = Path(self.data)
        return d if d.is_absolute() else self.repo / d


_settings: Settings | None = None


def settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Re-read the environment. Used by tests that set variables after import."""
    global _settings
    _settings = Settings()
    return _settings
