"""
Mission-layer settings, from the environment, read once per process.

Topic names come from :class:`fly_brain.config.Settings` so the two processes cannot disagree
about them. Everything here is ``MISSION_*``, ``MUSE_*``, ``CAMERA_DOWN_*`` or ``SIM_*``; never
``FLYCNS_*`` (see fly_brain.config for why).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from fly_brain.config import Settings, settings as brain_settings

HERE = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _f(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _i(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _b(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in ('1', 'true', 'yes', 'on')


def _xy(name: str, default: str) -> tuple[float, float]:
    a, b = (float(v) for v in _env(name, default).split(',')[:2])
    return a, b


def _xyz(name: str, default: str) -> tuple[float, float, float]:
    a, b, c = (float(v) for v in _env(name, default).split(',')[:3])
    return a, b, c


@dataclass
class MissionSettings:
    brain: Settings = field(default_factory=brain_settings)

    # --- how we reach the brain -----------------------------------------------
    link: str = field(default_factory=lambda: _env('MISSION_LINK', 'auto'))
    """``ros`` (topics to the fly_brain_pilot node), ``sim`` (in-process brain and kinematic
    world, for tests and demos without Gazebo) or ``auto`` (ros if rclpy imports, else sim)."""

    # --- the flight ------------------------------------------------------------
    altitude_m: float = field(default_factory=lambda: _f('MISSION_ALTITUDE_M', 15.0))
    """Flight level as world z. The forest floor spans about -2.5..+3 m, so 15 gives 12-17 m."""
    ground_z_m: float = field(default_factory=lambda: _f('MISSION_GROUND_Z_M', 0.0))
    speed_mps: float = field(default_factory=lambda: _f('MISSION_SPEED_MPS', 2.5))
    transit_speed_mps: float = field(default_factory=lambda: _f('MISSION_TRANSIT_SPEED_MPS', 4.0))
    arrive_radius_m: float = field(default_factory=lambda: _f('MISSION_ARRIVE_RADIUS_M', 1.5))
    transit_arrive_radius_m: float = field(
        default_factory=lambda: _f('MISSION_TRANSIT_ARRIVE_RADIUS_M', 2.5)
    )
    area_half_size_m: float = field(default_factory=lambda: _f('MISSION_AREA_HALF_SIZE_M', 58.0))
    """Default search area is the square [-h, h]^2: the 120 x 120 m forest map, less a margin."""
    lane_overlap: float = field(default_factory=lambda: _f('MISSION_LANE_OVERLAP', 0.2))
    keepout_clearance_m: float = field(default_factory=lambda: _f('MISSION_KEEPOUT_CLEARANCE_M', 4.0))
    keepout_vertical_clearance_m: float = field(
        default_factory=lambda: _f('MISSION_KEEPOUT_VERTICAL_CLEARANCE_M', 3.0)
    )
    keepouts_file: Path = field(
        default_factory=lambda: Path(_env('MISSION_KEEPOUTS', str(HERE / 'keepouts.json')))
    )
    sample_fraction: float = field(default_factory=lambda: _f('MISSION_SAMPLE_FRACTION', 0.5))
    """Camera samples every this fraction of the along-track footprint: consecutive frames overlap."""
    min_confidence: float = field(default_factory=lambda: _f('MISSION_MIN_CONFIDENCE', 0.6))
    confirm_detections: bool = field(default_factory=lambda: _b('MISSION_CONFIRM', True))
    leg_timeout_factor: float = field(default_factory=lambda: _f('MISSION_LEG_TIMEOUT_FACTOR', 3.0))
    leg_timeout_min_s: float = field(default_factory=lambda: _f('MISSION_LEG_TIMEOUT_MIN_S', 20.0))
    poll_s: float = field(default_factory=lambda: _f('MISSION_POLL_S', 0.05))
    max_frame_age_s: float = field(default_factory=lambda: _f('MISSION_MAX_FRAME_AGE_S', 2.0))

    # --- the downward camera -----------------------------------------------------
    cam_hfov_deg: float = field(default_factory=lambda: _f('CAMERA_DOWN_HFOV_DEG', 80.0))
    cam_width: int = field(default_factory=lambda: _i('CAMERA_DOWN_WIDTH', 640))
    cam_height: int = field(default_factory=lambda: _i('CAMERA_DOWN_HEIGHT', 480))

    # --- the detector -------------------------------------------------------------
    detector: str = field(default_factory=lambda: _env('PERSON_DETECTOR', 'auto'))
    """``muse`` (Meta Model API), ``color`` (an HSV blob, for the sim), ``none``, or ``auto``:
    muse if a key is set, else color."""
    muse_api_key: str = field(
        default_factory=lambda: _env('MUSE_API_KEY', '') or _env('MODEL_API_KEY', '')
    )
    muse_base_url: str = field(default_factory=lambda: _env('MUSE_BASE_URL', 'https://api.meta.ai/v1'))
    muse_model: str = field(default_factory=lambda: _env('MUSE_MODEL', 'muse-spark-1.3'))
    muse_timeout_s: float = field(default_factory=lambda: _f('MUSE_TIMEOUT_S', 60.0))
    muse_max_image_w: int = field(default_factory=lambda: _i('MUSE_MAX_IMAGE_W', 768))
    muse_reasoning_effort: str = field(default_factory=lambda: _env('MUSE_REASONING_EFFORT', 'low'))
    detect_color: str = field(default_factory=lambda: _env('DETECT_COLOR', 'magenta'))

    # --- the in-process sim link ---------------------------------------------------
    sim_person_xy: tuple[float, float] = field(default_factory=lambda: _xy('SIM_PERSON_XY', '2,-38'))
    """Where the kinematic sim puts the person. Default is person_target in forest_search.world.
    The planner never sees this; only the synthetic camera does."""
    sim_start: tuple[float, float, float] = field(default_factory=lambda: _xyz('SIM_START', '-9,-7,2.5'))
    """Where the kinematic sim starts: the X3's spawn pose in forest_search.world."""
    sim_speedup: float = field(default_factory=lambda: _f('SIM_SPEEDUP', 1.0))
    sim_dt: float = field(default_factory=lambda: _f('SIM_DT', 0.05))

    def default_area(self):
        from .schemas import Area

        h = self.area_half_size_m
        return Area.box(-h, -h, h, h)


_settings: MissionSettings | None = None


def mission_settings() -> MissionSettings:
    global _settings
    if _settings is None:
        _settings = MissionSettings()
    return _settings


def reload_mission_settings() -> MissionSettings:
    """Re-read the environment. For tests that set variables after import."""
    global _settings
    _settings = MissionSettings()
    return _settings
