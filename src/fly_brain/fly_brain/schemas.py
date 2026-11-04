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
Output = Literal['gf', 'jump', 'flight_power', 'steer', 'forward', 'courtship']
MotorState = Literal['HOLD', 'NAVIGATE', 'PURSUE', 'AVOID', 'ESCAPE', 'COOLDOWN']


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
    moving: bool | None = Field(
        default=None,
        description='Whether the object moves in the world, if the sensor can tell (the lidar '
        'tracks in the world frame). False is a stationary object: it never looms, but the '
        'motor analyses it for an up/down/sideways pass (fly_brain.dodge.choose_static_avoid). '
        'None: unknown (a camera cannot tell).',
    )
    height_m: float | None = Field(
        default=None,
        description='Vertical extent of the object, m, if measured (the 3-D lidar cluster '
        'height). size_deg is a width for lidar and must not be projected as a height.',
    )
    velocity_body: tuple[float, float, float] | None = Field(
        default=None,
        description="The object's own velocity, m/s, in body FLU, if a sensor tracks it (the "
        'lidar does). None: unknown, assumed straight at the drone.',
    )
    sound_db: float = Field(default=-80.0, description='Band-limited sound level, dBFS.')
    arousal: float = Field(
        default=0.0,
        description="Reproductive/approach gate 0..1. Muse raises it when it chooses to investigate "
        'a candidate; the pC1 -> DNp13 path then supplies a commitment signal, not a heading.',
    )
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


DodgeAxis = Literal['lateral', 'vertical']
DodgeDirection = Literal['left', 'right', 'up', 'down']


class DodgeAdvice(BaseModel):
    """
    Which way to get out of an object's path: the escape's direction, decided before it fires.

    Produced by the geometric predictor in :mod:`fly_brain.dodge` from the object's tracked
    motion, or by Muse Spark looking at the camera frame (``fly_drone.dodge_advisor``) when a
    key is set and the answer arrives in time. A giant-fibre spike then commits the drone to
    this direction plus a smaller push away from the object.
    """

    axis: DodgeAxis = Field(description="'lateral' (left/right) or 'vertical' (up/down).")
    direction: DodgeDirection = Field(description='Which way the drone goes.')
    expected: str = Field(default='', description="Where the object is expected to go: 'straight at us', 'passing on our left', ...")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: Literal['muse', 'geometry', 'default'] = Field(default='geometry')
    reason: str = Field(default='')
    track_key: str = Field(default='', description="The threat this is about, 'source:id'; '' is any.")
    t: float = Field(default=0.0, description='Wall clock it was decided, s since epoch.')


class ReflexParams(BaseModel):
    """Escape reflex tuning. A giant-fibre spike is a trigger EVENT, not a rate."""

    enabled: bool = Field(default=True, description='Whether the escape reflex may fire.')
    speed_mps: float = Field(default=3.0, description='Escape speed, m/s.')
    climb_deg: float = Field(default=35.0, description='Pitch of a vertical dodge above horizontal, degrees.')
    dodge_prefer: Literal['auto', 'lateral', 'vertical'] = Field(
        default='auto',
        description="'auto' lets the predictor choose: sideways for anything coming straight "
        'at the drone or from above or below, vertical only for a level crossing from the side. '
        "'lateral' or 'vertical' forces the axis.",
    )
    away_fraction: float = Field(
        default=0.45,
        description='Weight of the push straight away from the object, relative to the sideways '
        'dodge, in the escape vector. 0 is a pure sidestep; the old reflex was a pure retreat.',
    )
    advice_max_age_s: float = Field(
        default=6.0, description="How long a Muse dodge advice stays valid for its track."
    )
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
    dodge: DodgeAdvice | None = Field(default=None, description='Which way, and who decided.')


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
    roll_rad: float = Field(
        default=0.0, description='Roll, rad, about body x: + is left wing up (right wing down).'
    )
    pitch_rad: float = Field(
        default=0.0, description='Pitch, rad, about body y (left): + is nose DOWN, as the '
        'right-hand rule has it in FLU, the opposite of the aerospace convention.'
    )
    altitude_m: float = Field(default=0.0, description='Height above ground, m.')
    state: MotorState = Field(default='HOLD', description='Current behaviour state.')
    last_command: Command | None = Field(default=None, description='Most recent command computed.')
    ticks: int = Field(default=0, description='Control ticks executed.')


class BrainStatus(BaseModel):
    """What the brain is and how fast it runs."""

    backend: str = Field(description="'connectome' or 'mock'.")
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


class Goal(BaseModel):
    """
    A world-frame position for the brain to fly to. The mission layer's one message.

    The mission layer never commands velocity. It says *where*, the brain decides *how*: the
    escape and avoid reflexes outrank navigation, so an obstacle on the way produces a
    manoeuvre and then the drone resumes closing on the goal.
    """

    x: float = Field(description='World x (ENU east), m.')
    y: float = Field(description='World y (ENU north), m.')
    z: float = Field(description='World z (up), m. A flight level, not height above ground.')
    speed_mps: float | None = Field(
        default=None, description='Speed cap for this leg, m/s. None uses NavParams.speed_mps.'
    )
    arrive_radius_m: float | None = Field(
        default=None, description='Horizontal radius that counts as arrived. None uses NavParams.'
    )
    id: str = Field(default='', description='Caller-chosen id, echoed in GoalStatus.')
    label: str = Field(default='', description='Human-readable note, e.g. "lane 3 end".')
    arousal: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description='Reproductive/approach drive for this goal, 0..1. Muse sets it when the goal '
        'came from an investigate decision; it raises pC1 activity in the connectome, whose '
        'DNp13 readout is reported as courtship_drive. It does not steer.',
    )


class NavParams(BaseModel):
    """Goal navigation: position error to body-frame velocity, under the reflexes."""

    enabled: bool = Field(default=True, description='Whether a goal may command velocity.')
    speed_mps: float = Field(default=2.5, description='Default horizontal speed cap, m/s.')
    kp: float = Field(
        default=0.6,
        description='Horizontal P gain, (m/s)/m. The drone slows inside speed/kp of the goal.',
    )
    arrive_radius_m: float = Field(default=1.5, description='Default arrival radius, m.')
    arrive_dz_m: float = Field(default=1.0, description='Vertical tolerance for arrival, m.')
    altitude_gain: float = Field(default=0.8, description='Vertical P gain, (m/s)/m.')
    climb_first_dz_m: float = Field(
        default=3.0,
        description='When this far BELOW the goal, climb before translating: never cross the '
        'forest at tree height to save time.',
    )
    face_travel: bool = Field(
        default=True,
        description='Yaw to face the direction of travel so the forward camera, and with it '
        'the loom reflex, looks where the drone is going.',
    )
    yaw_gain: float = Field(default=1.2, description='rad/s per rad of heading error.')
    heading_min_speed_mps: float = Field(
        default=0.3, description='Below this speed the heading is left alone.'
    )


class GoalStatus(BaseModel):
    """How the brain is doing against its goal. Published every tick by the ROS node."""

    active: bool = Field(default=False, description='Whether a goal is set.')
    reached: bool = Field(default=False, description='Latched true once inside the arrival box.')
    goal: Goal | None = Field(default=None, description='The goal in force.')
    distance_m: float = Field(default=0.0, description='Straight-line distance to the goal, m.')
    horizontal_m: float = Field(default=0.0, description='Horizontal distance to the goal, m.')
    dz_m: float = Field(default=0.0, description='Goal z minus current z, m.')
    eta_s: float | None = Field(default=None, description='Rough time to arrive at the speed cap.')
    state: MotorState = Field(
        default='HOLD', description='Behaviour that produced the last command.'
    )
    position: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description='World x, y, z.'
    )
    yaw_rad: float = Field(default=0.0, description='Heading, rad.')
    connected: bool = Field(default=False, description='Whether odometry has arrived.')
    output_enabled: bool = Field(default=False, description='Whether commands reach the vehicle.')
    set_at: float | None = Field(default=None, description='Wall clock the goal was set.')
    reached_at: float | None = Field(default=None, description='Wall clock it was first reached.')
    courtship_drive: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description='pC1/DNp13 commitment readout, 0..1. It multiplies approach speed; it is '
        'not a heading. Muse chooses the goal.',
    )
    reproductive_triggered: bool = Field(
        default=False, description='Whether the reproductive pathway fired this tick.'
    )
    ticks: int = Field(default=0, description='Control ticks executed so far.')
