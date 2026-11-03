"""Which way to dodge: the predictor, the escape vector, and Muse's advice taking precedence."""
from __future__ import annotations

import math

from fly_brain.dodge import choose_dodge, closest_approach, escape_vector, relative_state
from fly_brain.motor import MotorLayer
from fly_brain.schemas import BrainOut, DodgeAdvice, DroneState, Percept


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
