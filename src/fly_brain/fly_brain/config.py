"""
Environment-driven settings.

Runtime data is vendored under ``fly_brain/data``; there is no clone path and no
repository setting.  A top-level ``.env`` is loaded with setdefault semantics, so
a key in the file is available to the MCP server and the ROS node without a
shell wrapper, while an explicitly exported variable still wins.
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


def _load_dotenv() -> str | None:
    """Load the repository's `.env` once, without overwriting exported values."""
    here = Path(__file__).resolve()
    for root in (here.parents[3], Path.cwd()):
        p = root / '.env'
        if not p.is_file():
            continue
        try:
            for raw in p.read_text(encoding='utf-8').splitlines():
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                if line.startswith('export '):
                    line = line[7:].strip()
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if value[:1] == value[-1:] and value[:1] in ('"', "'"):
                    value = value[1:-1]
                if key and value:
                    os.environ.setdefault(key, value)
            return str(p)
        except OSError:
            continue
    return None


ENV_FILE = _load_dotenv()


@dataclass
class Settings:
    """Everything tunable from the environment, read once per process."""

    # --- brain ---------------------------------------------------------------
    backend: str = field(default_factory=lambda: _env('BRAIN_BACKEND', 'connectome'))
    connectome_path: str = field(
        default_factory=lambda: _env(
            'FLYDRONE_CONNECTOME',
            str(Path(__file__).with_name('data') / 'hemibrain_subset.json.gz'),
        )
    )
    circuit_path: str = field(default_factory=lambda: _env('FLYDRONE_CIRCUIT', ''))
    brain_tick_ms: int = field(default_factory=lambda: _i('BRAIN_TICK_MS', 50))

    # --- control loop --------------------------------------------------------
    loop_hz: float = field(default_factory=lambda: _f('LOOP_HZ', 20.0))

    # --- sensors -------------------------------------------------------------
    camera_hfov_deg: float = field(default_factory=lambda: _f('CAMERA_HFOV_DEG', 60.0))
    lidar_enabled: bool = field(
        default_factory=lambda: _env('FLYDRONE_LIDAR', '1').strip().lower() in ('1', 'true', 'yes', 'on')
    )
    lidar_max_range_m: float = field(default_factory=lambda: _f('FLYDRONE_LIDAR_RANGE_M', 20.0))
    lidar_max_below_m: float = field(default_factory=lambda: _f('FLYDRONE_LIDAR_MAX_BELOW_M', 2.0))
    lidar_max_above_m: float = field(default_factory=lambda: _f('FLYDRONE_LIDAR_MAX_ABOVE_M', 4.0))

    # --- safety envelope -----------------------------------------------------
    max_speed: float = field(default_factory=lambda: _f('MAX_SPEED', 5.0))
    max_climb: float = field(default_factory=lambda: _f('MAX_CLIMB', 2.0))
    max_yaw: float = field(default_factory=lambda: _f('MAX_YAW', 2.0))
    min_altitude_m: float = field(default_factory=lambda: _f('MIN_ALTITUDE_M', 1.0))

    # --- ROS 2 interface -----------------------------------------------------
    ros_bridge: str = field(default_factory=lambda: _env('ROS_BRIDGE', 'twist'))
    topic_cmd_vel: str = field(
        default_factory=lambda: _env('ROS_TOPIC_CMD_VEL', '/drone/drone_motion')
    )
    topic_odom: str = field(default_factory=lambda: _env('ROS_TOPIC_ODOM', '/drone/odometry'))
    topic_image: str = field(default_factory=lambda: _env('ROS_TOPIC_IMAGE', '/drone/camera_image'))
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

    topic_goal: str = field(default_factory=lambda: _env('ROS_TOPIC_GOAL', '/fly_brain/goal'))
    topic_goal_status: str = field(
        default_factory=lambda: _env('ROS_TOPIC_GOAL_STATUS', '/fly_brain/goal_status')
    )

    # --- goal navigation -----------------------------------------------------
    nav_speed_mps: float = field(default_factory=lambda: _f('NAV_SPEED_MPS', 2.5))
    nav_arrive_radius_m: float = field(default_factory=lambda: _f('NAV_ARRIVE_RADIUS_M', 1.5))

    def connectome_file(self) -> Path:
        return Path(self.connectome_path).expanduser()


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
