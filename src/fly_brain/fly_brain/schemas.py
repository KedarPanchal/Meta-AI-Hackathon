"""
Pydantic models. These double as the MCP tool schemas, so the field docs are the tool docs.

Conventions, used everywhere:

* **Body frame is FLU**: ``x`` forward, ``y`` left, ``z`` up.
* ``yaw_rate`` is positive **counter-clockwise seen from above** (positive = turn left).
* ``bearing_deg`` is positive to the **RIGHT** of the nose. A target at ``bearing_deg=+30``
  needs a negative ``yaw_rate`` to centre it.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Channel = Literal['loom_speed', 'loom_size', 'target', 'sound', 'arousal']
Output = Literal['gf', 'jump', 'flight_power', 'steer', 'forward']
MotorState = Literal['HOLD', 'PURSUE', 'AVOID', 'ESCAPE', 'COOLDOWN']


class Box(BaseModel):
    """One detected object in the camera image, in pixels."""

    x: float = Field(description='Left edge, pixels.')
    y: float = Field(description='Top edge, pixels.')
    w: float = Field(description='Width, pixels.')
    h: float = Field(description='Height, pixels.')
    track_id: int = Field(default=-1, description='Stable id across frames; -1 if untracked.')
    label: str = Field(default='', description='Detector class name, if any.')
    image_w: float = Field(default=640.0, description='Image width, pixels.')
    image_h: float = Field(default=480.0, description='Image height, pixels.')


class Percept(BaseModel):
    """What the brain is told about the world on one tick."""

    bearing_deg: float = Field(default=0.0, description='Target bearing; + is RIGHT of the nose.')
    elevation_deg: float = Field(default=0.0, description='Target elevation; + is UP.')
    size_deg: float = Field(default=0.0, description='Angular size of the object.')
    expansion_dps: float = Field(
        default=0.0,
        description='Angular size rate of change, deg/s (loom).',
    )
    distance_m: float | None = Field(default=None, description='Range, if anything measures it.')
    sound_db: float = Field(default=-80.0, description='Band-limited sound level, dBFS.')
    arousal: float = Field(default=0.0, description="Mission gate, 0..1: the 'pursue now' drive.")
    altitude_m: float = Field(default=0.0, description='Height above ground, m.')
    has_target: bool = Field(default=False, description='Whether a target was seen this tick.')


class Drive(BaseModel):
    """Poisson rates handed to the brain, per channel, as ``[left, right]`` in Hz."""

    rates: dict[str, tuple[float, float]] = Field(
        default_factory=dict, description='channel -> (left_hz, right_hz).'
    )


class BrainOut(BaseModel):
    """Spike counts read back from the monitored output types for one tick."""

    tick_ms: float = Field(description='Simulated duration of this tick, ms.')
    spikes: dict[str, int] = Field(
        default_factory=dict,
        description='output channel -> total spikes.',
    )
    spikes_lr: dict[str, tuple[int, int]] = Field(
        default_factory=dict, description='output channel -> (left, right) spikes.'
    )
    wall_ms: float = Field(default=0.0, description='Wall-clock time this tick cost, ms.')


class Command(BaseModel):
    """The velocity command sent to the vehicle. Body frame FLU."""

    vx: float = Field(default=0.0, description='Forward velocity, m/s.')
    vy: float = Field(default=0.0, description='Left velocity, m/s.')
    vz: float = Field(default=0.0, description='Up velocity, m/s.')
    yaw_rate: float = Field(default=0.0, description='Yaw rate, rad/s; + is left (CCW).')
    state: MotorState = Field(default='HOLD', description='Which behaviour produced this command.')
    reason: str = Field(default='', description='Human-readable note on why.')


class ReflexParams(BaseModel):
    """Escape reflex tuning. A giant-fibre spike is a trigger EVENT, not a rate."""

    enabled: bool = Field(default=True, description='Whether the escape reflex may fire.')
    speed_mps: float = Field(default=3.0, description='Escape speed, m/s.')
    climb_deg: float = Field(default=35.0, description='Escape pitch above horizontal, degrees.')
    per_extra_spike: float = Field(
        default=0.5,
        description='Speed multiplier added per extra GF spike.',
    )
    psi_bonus: float = Field(default=0.25, description='Extra speed fraction if PSI also spiked.')
    hold_ms: float = Field(default=350.0, description='Full-speed escape duration, ms.')
    decay_ms: float = Field(default=250.0, description='Ramp-down after the hold, ms.')
    cooldown_ms: float = Field(default=800.0, description='Refractory window after an escape, ms.')
    pursuit_damping: float = Field(
        default=0.3,
        description='Pursuit gain multiplier during cooldown.',
    )
    direction_source: Literal['vision', 'brain'] = Field(
        default='vision',
        description=(
            "Where the escape heading comes from. 'vision' uses the threat bearing. "
            "'brain' uses the left/right GF imbalance, which on the toy graph points "
            'TOWARD the threat, because loom input reaches both giant fibres.'
        ),
    )
    accept_ttmn: bool = Field(
        default=True,
        description='Treat a TTMn (jump motor) spike as a trigger too.',
    )
    avoid_speed_mps: float = Field(
        default=1.2,
        description='Graded side-step speed when a loom draws no GF spike.',
    )
    suppress_below_horizon: bool = Field(
        default=False,
        description='Ignore looms below the horizon (the ground expands while landing).',
    )


class PursuitParams(BaseModel):
    """Pursuit decoder: descending-neuron rates to (forward, yaw)."""

    enabled: bool = Field(default=True, description='Whether pursuit may command velocity.')
    arousal: float = Field(default=0.0, description='Tonic P1/pC1 gate, 0..1. 0 means hover.')
    yaw_gain: float = Field(
        default=0.0,
        description='rad/s per (left-right) spike imbalance. Set by calibration.',
    )
    yaw_bias: float = Field(default=0.0, description='rad/s offset from the yaw fit.')
    forward_gain: float = Field(
        default=0.0,
        description='m/s per forward-neuron spike. Gain-only on purpose: with no intercept, '
        'arousal=0 makes the drone hover.',
    )
    stop_distance_m: float = Field(default=1.5, description='Stop closing inside this range.')
    cruise_altitude_m: float = Field(
        default=3.0,
        description='Altitude the classic controller holds, m.',
    )
    altitude_gain: float = Field(default=0.8, description='Vertical P gain, (m/s)/m.')


class ReflexEvent(BaseModel):
    """One logged reflex firing."""

    t: float = Field(description='Wall clock, seconds since epoch.')
    kind: Literal['ESCAPE', 'AVOID'] = Field(description='Which reflex fired.')
    gf_spikes: int = Field(default=0, description='Giant-fibre spikes in the triggering tick.')
    ttmn_spikes: int = Field(default=0, description='Jump motor-neuron spikes.')
    psi_spikes: int = Field(default=0, description='Flight-power (PSI) spikes.')
    bearing_deg: float = Field(default=0.0, description='Threat bearing at the trigger.')
    size_deg: float = Field(default=0.0, description='Threat angular size at the trigger.')
    expansion_dps: float = Field(default=0.0, description='Threat expansion rate at the trigger.')
    distance_m: float | None = Field(default=None, description='Threat range, if known.')
    ttc_s: float | None = Field(default=None, description='Time to contact at the trigger, s.')
    command: Command | None = Field(default=None, description='The command the reflex produced.')


class DroneState(BaseModel):
    """Last known vehicle state and what the controller last decided."""

    connected: bool = Field(default=False, description='Whether odometry has ever arrived.')
    output_enabled: bool = Field(
        default=False,
        description='Whether commands actually reach the vehicle.',
    )
    position: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description='World x, y, z (ENU), m.',
    )
    velocity: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description='World vx, vy, vz, m/s.',
    )
    yaw_rad: float = Field(default=0.0, description='Heading, rad.')
    altitude_m: float = Field(default=0.0, description='Height above ground, m.')
    state: MotorState = Field(default='HOLD', description='Current behaviour state.')
    last_command: Command | None = Field(default=None, description='Most recent command computed.')
    ticks: int = Field(default=0, description='Control ticks executed.')


class BrainStatus(BaseModel):
    """What the brain is and how fast it runs."""

    backend: str = Field(description="'mock' or 'flycns'.")
    loaded: bool = Field(default=False, description='Whether the network is built.')
    n_neurons: int = Field(default=0, description='Simulated LIF neurons.')
    n_edges: int = Field(default=0, description='Signed chemical edges.')
    n_stim: int = Field(default=0, description='Neurons replaced by Poisson sources.')
    data_dir: str = Field(default='', description='Graph directory in use.')
    tick_ms: int = Field(default=50, description='Simulated ms per control tick.')
    realtime_factor: float = Field(
        default=0.0,
        description=(
            'Simulated time / wall time over recent ticks. '
            'Below 1.0 is slower than real time.'
        ),
    )
    missing_types: list[str] = Field(
        default_factory=list,
        description='Requested stimulus types not in the graph.',
    )
    ablated: list[str] = Field(default_factory=list, description='Types whose edges were dropped.')
    ticks: int = Field(default=0, description='Brain ticks run.')
