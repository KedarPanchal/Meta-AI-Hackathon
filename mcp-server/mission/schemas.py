"""Pydantic models. They double as the MCP tool schemas, so the field docs are the tool docs."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from fly_brain.schemas import GoalStatus

Point2 = tuple[float, float]

MissionMode = Literal['search', 'scout']
MissionState = Literal['idle', 'planned', 'running', 'found', 'complete', 'aborted', 'failed']


class Area(BaseModel):
    """Where to look: an axis-aligned box or a polygon, in world x/y (ENU, metres)."""

    x_min: float | None = Field(default=None, description='Box west edge, m.')
    x_max: float | None = Field(default=None, description='Box east edge, m.')
    y_min: float | None = Field(default=None, description='Box south edge, m.')
    y_max: float | None = Field(default=None, description='Box north edge, m.')
    polygon: list[Point2] | None = Field(
        default=None,
        description='Polygon vertices [[x, y], ...] instead of a box. Any winding; need not be convex.',
    )

    @model_validator(mode='after')
    def _one_shape(self) -> 'Area':
        box = [self.x_min, self.x_max, self.y_min, self.y_max]
        if self.polygon is not None:
            if len(self.polygon) < 3:
                raise ValueError('polygon needs at least 3 vertices')
            if any(v is not None for v in box):
                raise ValueError('give a polygon OR a box, not both')
        elif any(v is None for v in box):
            raise ValueError('give all four of x_min, x_max, y_min, y_max, or a polygon')
        elif self.x_min >= self.x_max or self.y_min >= self.y_max:  # type: ignore[operator]
            raise ValueError('x_min < x_max and y_min < y_max required')
        return self

    @classmethod
    def box(cls, x_min: float, y_min: float, x_max: float, y_max: float) -> 'Area':
        return cls(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    def vertices(self) -> list[Point2]:
        if self.polygon is not None:
            return [(float(x), float(y)) for x, y in self.polygon]
        assert self.x_min is not None and self.x_max is not None
        assert self.y_min is not None and self.y_max is not None
        return [
            (self.x_min, self.y_min), (self.x_max, self.y_min),
            (self.x_max, self.y_max), (self.x_min, self.y_max),
        ]


class Keepout(BaseModel):
    """
    Something the path must route around: a circle or a polygon, and how high it reaches.

    Only keep-outs whose ``top_z_m`` comes within the vertical clearance of the flight level are
    obstacles; anything lower is just terrain under the camera. Trees are deliberately not
    keep-outs: the drone flies above the canopy and unexpected ones are the brain's problem.
    """

    name: str = Field(default='', description='For the log.')
    x: float | None = Field(default=None, description='Circle centre x, m.')
    y: float | None = Field(default=None, description='Circle centre y, m.')
    radius_m: float | None = Field(default=None, description='Circle radius, m.')
    polygon: list[Point2] | None = Field(default=None, description='Polygon instead of a circle.')
    top_z_m: float = Field(
        default=1e9,
        description='World z of the top of the structure. Default: always an obstacle.',
    )

    @model_validator(mode='after')
    def _one_shape(self) -> 'Keepout':
        circle = self.x is not None and self.y is not None and self.radius_m is not None
        if self.polygon is None and not circle:
            raise ValueError('a keepout needs x, y, radius_m or a polygon')
        if self.polygon is not None and len(self.polygon) < 3:
            raise ValueError('polygon needs at least 3 vertices')
        return self

    def footprint(self, clearance_m: float, n: int = 12) -> list[Point2]:
        """The region to stay out of, grown by the horizontal clearance."""
        from . import geometry

        if self.polygon is not None:
            return geometry.buffer_polygon([(float(x), float(y)) for x, y in self.polygon], clearance_m)
        assert self.x is not None and self.y is not None and self.radius_m is not None
        return geometry.circle_polygon(self.x, self.y, self.radius_m + clearance_m, n=n)

    def blocks_level(self, altitude_m: float, vertical_clearance_m: float) -> bool:
        return self.top_z_m + vertical_clearance_m >= altitude_m


class Waypoint(BaseModel):
    """One goal the mission layer will hand the brain."""

    x: float
    y: float
    z: float
    kind: Literal['transit', 'lane_start', 'lane_end'] = 'transit'
    cell: int = Field(default=-1, description='Decomposition cell index, -1 for transits.')
    lane: int = Field(default=-1, description='Lane index within the mission, -1 for transits.')
    speed_mps: float | None = None
    arrive_radius_m: float | None = None

    def xy(self) -> Point2:
        return (self.x, self.y)

    def dist_to(self, other: 'Waypoint') -> float:
        return math.hypot(other.x - self.x, other.y - self.y)


class Cell(BaseModel):
    """One cell of the boustrophedon decomposition."""

    index: int
    polygon: list[Point2]
    n_lanes: int
    area_m2: float


class CoveragePlan(BaseModel):
    """The output of the planner: cells, lanes, and the ordered goals that fly them."""

    area: list[Point2] = Field(description='Search polygon actually used.')
    keepouts: list[list[Point2]] = Field(default_factory=list, description='Obstacle footprints used.')
    keepouts_ignored: list[str] = Field(
        default_factory=list, description='Keep-outs below the flight level, so not routed around.'
    )
    altitude_m: float = Field(description='Flight level, world z.')
    height_above_ground_m: float = Field(description='Assumed camera height for the footprint.')
    footprint_w_m: float = Field(description='Camera footprint across-track, m.')
    footprint_l_m: float = Field(description='Camera footprint along-track, m.')
    lane_spacing_m: float
    lane_overlap: float
    sweep_heading_deg: float = Field(description='Direction the lanes run, degrees from +x (east).')
    cells: list[Cell]
    waypoints: list[Waypoint]
    n_lanes: int
    lane_length_m: float
    transit_length_m: float
    path_length_m: float
    est_duration_s: float
    free_area_m2: float
    coverage_estimate: float = Field(description='Fraction of free area inside some footprint, 0..1.')
    sample_spacing_m: float = Field(description='Metres between camera samples along the path.')
    est_samples: int = Field(description='Roughly how many detector calls the mission will make.')
    warnings: list[str] = Field(default_factory=list)

    def summary(self) -> dict:
        return {
            'n_cells': len(self.cells),
            'n_lanes': self.n_lanes,
            'n_waypoints': len(self.waypoints),
            'path_length_m': round(self.path_length_m, 1),
            'lane_length_m': round(self.lane_length_m, 1),
            'transit_length_m': round(self.transit_length_m, 1),
            'est_duration_s': round(self.est_duration_s, 0),
            'altitude_m': self.altitude_m,
            'footprint_m': [round(self.footprint_w_m, 1), round(self.footprint_l_m, 1)],
            'lane_spacing_m': round(self.lane_spacing_m, 2),
            'sweep_heading_deg': round(self.sweep_heading_deg, 1),
            'free_area_m2': round(self.free_area_m2, 0),
            'coverage_estimate': round(self.coverage_estimate, 3),
            'sample_spacing_m': round(self.sample_spacing_m, 1),
            'est_samples': self.est_samples,
            'keepouts_used': len(self.keepouts),
            'keepouts_ignored': self.keepouts_ignored,
            'warnings': self.warnings,
        }


class Detection(BaseModel):
    """What the detector said about one frame."""

    person_visible: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description='Normalised [x0, y0, x1, y1], origin top-left, 0..1.'
    )
    description: str = ''
    detector: str = ''
    latency_ms: float = 0.0
    error: str | None = Field(default=None, description='Set when the detector failed; then trust nothing else.')
    raw: str | None = Field(default=None, description='The model text, truncated, for debugging.')

    def centre(self) -> Point2 | None:
        if self.bbox is None:
            return None
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


class Sighting(BaseModel):
    """A positive detection placed in the world."""

    t: float = Field(description='Wall clock, s since epoch.')
    detection: Detection
    drone_position: tuple[float, float, float]
    drone_yaw_rad: float
    estimated_xy: Point2 | None = Field(
        default=None,
        description='Ground position of the person from the bbox centre, flat-ground assumption.',
    )
    waypoint_index: int = -1
    confirmed: bool = Field(default=False, description='Seen again from directly above.')
    n_sightings: int = Field(default=1, description='Positive frames merged into this sighting.')


class MissionStatus(BaseModel):
    """Everything an agent needs to follow a mission."""

    id: str | None = None
    mode: MissionMode | None = None
    state: MissionState = 'idle'
    link: str = ''
    detector: str = ''
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_s: float = 0.0
    waypoint_index: int = 0
    n_waypoints: int = 0
    progress: float = Field(default=0.0, description='Fraction of the planned path flown, 0..1.')
    distance_flown_m: float = 0.0
    position: tuple[float, float, float] | None = None
    yaw_rad: float | None = None
    brain: GoalStatus | None = Field(default=None, description="The brain's view of its goal.")
    frame_age_s: float | None = Field(default=None, description='Age of the newest camera frame.')
    samples: int = Field(default=0, description='Detector calls made.')
    detector_errors: int = 0
    last_detector_error: str | None = None
    sightings: list[Sighting] = Field(default_factory=list)
    best: Sighting | None = Field(default=None, description='Highest-confidence sighting so far.')
    plan: dict | None = Field(default=None, description='CoveragePlan.summary().')
    next_waypoints: list[Waypoint] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list, description='Most recent mission log lines.')
    note: str = ''
