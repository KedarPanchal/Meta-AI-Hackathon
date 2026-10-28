"""
Environment-driven settings.

Never name a setting ``FLYCNS_*``: FLYCNS's own ``announce_overrides()`` treats any such
variable as a declared perturbation of the model and stamps it into every report. Ours are
``FLYDRONE_*``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
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
    camera_hfov_deg: float = field(default_factory=lambda: _f('CAMERA_HFOV_DEG', 90.0))

    # --- safety envelope -----------------------------------------------------
    max_speed: float = field(default_factory=lambda: _f('MAX_SPEED', 5.0))
    max_climb: float = field(default_factory=lambda: _f('MAX_CLIMB', 2.0))
    max_yaw: float = field(default_factory=lambda: _f('MAX_YAW', 2.0))
    min_altitude_m: float = field(default_factory=lambda: _f('MIN_ALTITUDE_M', 1.0))

    # --- ROS 2 interface -----------------------------------------------------
    # Default target is plain Gazebo through ros_gz_bridge: a body-frame Twist.
    ros_bridge: str = field(default_factory=lambda: _env('ROS_BRIDGE', 'twist'))
    topic_cmd_vel: str = field(default_factory=lambda: _env('ROS_TOPIC_CMD_VEL', '/xuav/cmd_vel'))
    topic_odom: str = field(default_factory=lambda: _env('ROS_TOPIC_ODOM', '/xuav/odom'))
    topic_image: str = field(
        default_factory=lambda: _env('ROS_TOPIC_IMAGE', '/xuav/camera/image_raw')
    )
    topic_audio: str = field(default_factory=lambda: _env('ROS_TOPIC_AUDIO', '/xuav/audio'))
    frame_id: str = field(default_factory=lambda: _env('ROS_FRAME_ID', 'base_link'))

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
