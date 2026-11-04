"""Which way to dodge: the predictor, the escape vector, and Muse's advice taking precedence."""
from __future__ import annotations

import math

from fly_brain.config import Settings
from fly_brain.dodge import choose_dodge, choose_static_avoid, closest_approach, escape_vector, relative_state
from fly_brain.motor import MotorLayer
from fly_brain.schemas import BrainOut, DodgeAdvice, DroneState, Goal, Percept


def percept(bearing=0.0, elevation=0.0, distance=8.0, v=None) -> Percept:
    return Percept(has_target=True, bearing_deg=bearing, elevation_deg=elevation, size_deg=5.0,
                   expansion_dps=10.0, distance_m=distance, velocity_body=v)


def test_head_on_goes_sideways_not_up() -> None:
    d = choose_dodge(percept(v=(-5.0, 0.0, 0.0)))
    assert d.axis == 'lateral' and d.direction in ('left', 'right')
    assert 'head-on' in d.reason and 'straight at us' in d.expected


def test_unknown_velocity_is_assumed_straight_at_us() -> None:
    d = choose_dodge(percept(v=None))
    assert d.axis == 'lateral' and d.source == 'default'


def test_dodges_away_from_the_side_it_will_pass() -> None:
    # coming from ahead, drifting to our left: it will pass on our left, so we go right
    d = choose_dodge(percept(v=(-5.0, 0.4, 0.0)))
    assert d.direction == 'right'
    d = choose_dodge(percept(v=(-5.0, -0.4, 0.0)))
    assert d.direction == 'left'
    # already going to pass well to our right (miss 2.7 m): still go left, further from it
    d = choose_dodge(percept(bearing=20.0, v=(-5.0, 0.0, 0.0)))
    p, _ = relative_state(percept(bearing=20.0))
    miss, _ = closest_approach(p, (-5.0, 0.0, 0.0))
    assert miss[1] < -2.0 and d.direction == 'left' and 'passing to our right' in d.expected


def test_from_above_or_below_goes_sideways() -> None:
    d = choose_dodge(percept(elevation=50.0, v=(-3.0, 0.0, -4.0)))
    assert d.axis == 'lateral' and 'above/below' in d.reason and 'coming down' in d.expected
    d = choose_dodge(percept(elevation=-50.0, v=(-3.0, 0.0, 4.0)))
    assert d.axis == 'lateral'


def test_level_crossing_from_the_side_goes_vertical() -> None:
    # from 90 degrees right, flying left across our nose, at our level
    d = choose_dodge(percept(bearing=90.0, v=(0.0, 5.0, 0.0)))
    assert d.axis == 'vertical' and d.direction == 'up'
    # ...and down if it will pass above us
    d = choose_dodge(percept(bearing=90.0, elevation=5.0, v=(0.0, 5.0, 0.0)))
    assert d.direction == 'down'


def test_forced_axis_and_fresh_muse_advice_win() -> None:
    d = choose_dodge(percept(v=(-5.0, 0.0, 0.0)), prefer='vertical')
    assert d.axis == 'vertical'
    muse = DodgeAdvice(axis='lateral', direction='right', source='muse', confidence=0.8,
                       track_key='lidar:7', t=100.0, expected='a bird veering left')
    d = choose_dodge(percept(bearing=90.0, v=(0.0, 5.0, 0.0)), advice=muse, track_key='lidar:7', now=103.0)
    assert d is muse
    # stale, or about another track, or unsure: geometry again
    assert choose_dodge(percept(bearing=90.0, v=(0.0, 5.0, 0.0)), advice=muse, track_key='lidar:7', now=120.0).source != 'muse'
    assert choose_dodge(percept(bearing=90.0, v=(0.0, 5.0, 0.0)), advice=muse, track_key='lidar:8', now=101.0).source != 'muse'
    unsure = muse.model_copy(update={'confidence': 0.3})
    assert choose_dodge(percept(bearing=90.0, v=(0.0, 5.0, 0.0)), advice=unsure, track_key='lidar:7', now=101.0).source != 'muse'


def test_escape_vector_shapes() -> None:
    p = (8.0, 0.0, 0.0)                                             # dead ahead
    lat = DodgeAdvice(axis='lateral', direction='left')
    ux, uy, uz = escape_vector(p, lat, away_fraction=0.45, climb_deg=35.0)
    assert uy > 0.8 and ux < 0 and abs(uz) < 1e-9                   # left, a little back, no climb
    vert = DodgeAdvice(axis='vertical', direction='up')
    ux, uy, uz = escape_vector((0.0, -8.0, 0.0), vert, away_fraction=0.45, climb_deg=35.0)
    assert math.isclose(uz, math.sin(math.radians(35.0)), abs_tol=1e-6) and uy > 0.5


def static_percept(bearing=0.0, elevation=0.0, size=30.0, distance=3.0, altitude=2.5,
                   height=None) -> Percept:
    return Percept(has_target=True, bearing_deg=bearing, elevation_deg=elevation, size_deg=size,
                   expansion_dps=0.0, distance_m=distance, moving=False, altitude_m=altitude,
                   height_m=height)


# Stationary analysis geometry, at a 2.5 m hover unless stated: elevation/size
# describe the object's vertical span projected at its range (see dodge.py).

def test_static_desk_below_needs_no_manoeuvre() -> None:
    # a 0.75 m desk 3 m ahead of a 2.5 m hover: its top stays under the corridor
    assert choose_static_avoid(static_percept(elevation=-35.0, size=14.0, height=0.75), 2.5, 1.0, 2.5) is None
    # too far away, or beside the path (a column being passed), is navigation's
    # business as well: reacting to those shoves the drone into the other side
    assert choose_static_avoid(static_percept(distance=8.0), 2.5, 1.0, 2.5) is None
    assert choose_static_avoid(static_percept(bearing=90.0), 2.5, 1.0, 2.5) is None
    assert choose_static_avoid(static_percept(bearing=60.0, size=30.0), 2.5, 1.0, 2.5) is None


def test_static_wide_low_object_is_not_projected_tall() -> None:
    # a 53 deg-wide, 0.75 m-tall desk: size_deg is its width, so the old
    # size-as-height projection invented a ceiling-high obstacle; the real
    # height passes under the corridor and needs no manoeuvre
    assert choose_static_avoid(static_percept(elevation=-35.0, size=53.0, height=0.75), 2.5, 1.0, 2.5) is None


def test_static_low_obstacle_goes_up_and_over() -> None:
    # flying low at 1.2 m toward a 0..1.6 m obstacle: under it is the floor, over it is open
    d = choose_static_avoid(static_percept(elevation=-7.6, size=30.0, altitude=1.2, height=1.6), 1.2, 1.0, 0.0)
    assert d is not None and d.axis == 'vertical' and d.direction == 'up' and 'over it' in d.reason


def test_static_hanging_obstacle_goes_down_and_under() -> None:
    # a 2.7..4.0 m obstacle at the 2.5 m indoor level: the ceiling cap forbids over, under is open
    d = choose_static_avoid(static_percept(elevation=15.8, size=25.0, height=1.3), 2.5, 1.0, 2.5)
    assert d is not None and d.axis == 'vertical' and d.direction == 'down' and 'under it' in d.reason


def test_static_full_wall_indoors_goes_sideways() -> None:
    # a 0..2.9 m wall at the 2.5 m indoor level: over reaches the cap, under reaches the floor
    d = choose_static_avoid(static_percept(bearing=10.0, elevation=-19.3, size=51.6, height=2.9), 2.5, 1.0, 2.5)
    assert d is not None and d.axis == 'lateral' and d.direction == 'left'  # object right -> go left
    assert 'too tall to cross' in d.reason
    # without the indoor cap the same wall is simply flown over
    d = choose_static_avoid(static_percept(bearing=10.0, elevation=-19.3, size=51.6, height=2.9), 2.5, 1.0, 0.0)
    assert d is not None and d.direction == 'up'


def test_motor_static_avoid_outranks_navigation() -> None:
    out = BrainOut(tick_ms=50, spikes={})
    st = DroneState(connected=True, position=(0.0, 0.0, 2.5), altitude_m=2.5)
    # indoors, a wall ahead on the way to the goal: sidestep, do not pin against it
    m = MotorLayer(cfg=Settings(max_altitude_m=2.5))
    m.set_goal(Goal(x=10.0, y=0.0, z=2.5))
    cmd = m.decide(static_percept(bearing=10.0, elevation=-19.3, size=51.6, height=2.9), out, st, now=0.0)
    assert cmd.state == 'AVOID' and cmd.vy > 0.15 and cmd.vx < 0.0 and cmd.vz == 0.0
    assert 'stationary avoid left' in cmd.reason and m.events[-1].dodge.direction == 'left'
    # a low obstacle with no goal: climb over it, keeping the chosen climb through _clamp
    m2 = MotorLayer()
    cmd = m2.decide(static_percept(elevation=-7.6, size=30.0, altitude=1.2, height=1.6), out,
                    DroneState(connected=True, position=(0.0, 0.0, 1.2), altitude_m=1.2), now=0.0)
    assert cmd.state == 'AVOID' and cmd.vz > 0.25 and 'stationary avoid up' in cmd.reason
    # a desk below a goal-bound drone is flown over without any manoeuvre at all
    m3 = MotorLayer(cfg=Settings(max_altitude_m=2.5))
    m3.set_goal(Goal(x=10.0, y=0.0, z=2.5))
    cmd = m3.decide(static_percept(elevation=-35.0, size=14.0, height=0.75), out, st, now=0.0)
    assert cmd.state == 'NAVIGATE'


def test_motor_escape_is_a_sidestep_and_logs_the_dodge() -> None:
    m = MotorLayer()
    out = BrainOut(tick_ms=50, spikes={'gf': 1, 'jump': 1, 'flight_power': 0})
    st = DroneState(connected=True, position=(0.0, 0.0, 15.0), altitude_m=15.0)
    cmd = m.decide(percept(v=(-5.0, -0.5, 0.0)), out, st, now=0.0, track_key='lidar:1')
    assert cmd.state == 'ESCAPE' and cmd.vy > 1.5 and cmd.vz == 0.0 and cmd.vx < 0
    assert 'left' in cmd.reason and m.events[-1].dodge is not None
    assert m.events[-1].dodge.direction == 'left' and m.last_dodge is not None
    # Muse advice for this track turns the same escape into a climb
    m2 = MotorLayer()
    m2.set_advice(DodgeAdvice(axis='vertical', direction='up', source='muse', confidence=0.9,
                              track_key='lidar:1', t=0.0, expected='it will swerve left'))
    cmd = m2.decide(percept(v=(-5.0, -0.5, 0.0)), out, st, now=0.0, track_key='lidar:1')
    assert cmd.vz > 1.0 and 'muse' in cmd.reason
