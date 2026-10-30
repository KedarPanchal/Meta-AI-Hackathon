"""The brain's NAVIGATE behaviour: a goal becomes body-frame velocity, under the reflexes."""
from __future__ import annotations

import math
import time

import pytest

from fly_brain.schemas import Command, DroneState

from fly_brain.config import reload_settings
from fly_brain.controller import Controller
from fly_brain.schemas import Goal, Percept


def make_ctl() -> Controller:
    cfg = reload_settings()
    ctl = Controller(cfg)
    ctl.load_brain('mock')
    ctl.state.connected = True
    ctl.state.position = (0.0, 0.0, 15.0)
    ctl.state.altitude_m = 15.0
    ctl.state.yaw_rad = 0.0
    return ctl


def fly(ctl: Controller, seconds: float, dt: float = 0.05, percept: Percept | None = None):
    """Kinematic integration of the brain's commands, world ENU, no lag. Returns the commands."""
    cmds = []
    for _ in range(int(seconds / dt)):
        x, y, z = ctl.state.position
        yaw = ctl.state.yaw_rad
        ctl.state.altitude_m = z
        p = percept or Percept(has_target=False, altitude_m=z)
        cmd = ctl.tick(p)
        cmds.append(cmd)
        c, s = math.cos(yaw), math.sin(yaw)
        wx, wy = c * cmd.vx - s * cmd.vy, s * cmd.vx + c * cmd.vy
        ctl.state.position = (x + wx * dt, y + wy * dt, z + cmd.vz * dt)
        ctl.state.yaw_rad = yaw + cmd.yaw_rate * dt
    return cmds


def test_reaches_goal_and_latches() -> None:
    ctl = make_ctl()
    ctl.set_goal(Goal(x=20.0, y=-10.0, z=15.0, id='g1', label='test'))
    cmds = fly(ctl, 25.0)
    st = ctl.goal_status()
    assert st.active and st.reached, st
    assert st.goal is not None and st.goal.id == 'g1'
    assert st.horizontal_m < 1.6
    assert all(c.state == 'NAVIGATE' for c in cmds)
    # it slows into the goal rather than overshooting: the last commands are small
    assert math.hypot(cmds[-1].vx, cmds[-1].vy) < 0.6
    # and the latch holds even if the drone is pushed off the spot
    ctl.state.position = (24.0, -10.0, 15.0)
    fly(ctl, 0.2)
    assert ctl.goal_status().reached


def test_speed_cap_and_safety_clamp() -> None:
    ctl = make_ctl()
    ctl.set_goal(Goal(x=200.0, y=0.0, z=15.0, speed_mps=1.0))
    cmds = fly(ctl, 3.0)
    assert max(math.hypot(c.vx, c.vy) for c in cmds) <= 1.0 + 1e-9
    ctl.set_goal(Goal(x=200.0, y=0.0, z=15.0, speed_mps=50.0))
    cmds = fly(ctl, 1.0)
    assert max(math.hypot(c.vx, c.vy) for c in cmds) <= ctl.cfg.max_speed + 1e-9


def test_climb_first_then_translate() -> None:
    """From the ground the drone climbs before it moves: no crossing the forest at tree height."""
    ctl = make_ctl()
    ctl.state.position = (0.0, 0.0, 2.5)
    ctl.set_goal(Goal(x=40.0, y=0.0, z=15.0))
    cmds = fly(ctl, 12.0)
    zs = []
    z = 2.5
    for c in cmds:
        if z < 15.0 - ctl.motor.nav.climb_first_dz_m - 1e-6:
            assert c.vx == 0.0 and c.vy == 0.0, f'translated at z={z:.1f}: {c}'
            assert c.vz > 0
        z += c.vz * 0.05
        zs.append(z)
    assert ctl.state.position[0] > 5.0, 'never started moving once high enough'
    assert abs(ctl.state.position[2] - 15.0) < 1.0


def test_body_frame_conversion_uses_heading() -> None:
    """Facing north, a goal due east must be commanded to the RIGHT (negative vy), not forward."""
    ctl = make_ctl()
    ctl.motor.nav.face_travel = False
    ctl.state.yaw_rad = math.pi / 2.0
    ctl.set_goal(Goal(x=30.0, y=0.0, z=15.0))
    cmd = ctl.tick(Percept(has_target=False, altitude_m=15.0))
    assert cmd.vy < -1.0
    assert abs(cmd.vx) < 1e-6


def test_faces_the_direction_of_travel() -> None:
    ctl = make_ctl()
    ctl.set_goal(Goal(x=0.0, y=40.0, z=15.0))        # due north, drone faces east
    fly(ctl, 6.0)
    yaw = (ctl.state.yaw_rad + math.pi) % (2 * math.pi) - math.pi
    assert abs(yaw - math.pi / 2.0) < 0.2, f'yaw {yaw:.2f}, wanted pi/2'


def test_reflex_outranks_navigation() -> None:
    """A loom fires the escape in the middle of a leg, and navigation resumes after it."""
    ctl = make_ctl()
    ctl.set_goal(Goal(x=50.0, y=0.0, z=15.0))
    fly(ctl, 2.0)
    loom = Percept(has_target=True, bearing_deg=0.0, size_deg=40.0, expansion_dps=350.0, altitude_m=15.0)
    cmds = fly(ctl, 0.3, percept=loom)
    states = {c.state for c in cmds}
    assert 'ESCAPE' in states, states
    assert cmds[0].vx < 0, 'escape must back away from the head-on threat'
    # the escape clock runs on the wall clock (hold + decay + cooldown is 1.4 s), so wait it out
    time.sleep(1.6)
    later = fly(ctl, 1.0)
    assert later[-1].state == 'NAVIGATE', later[-1]
    assert ctl.goal_status().active


def test_goal_altitude_is_held_while_avoiding() -> None:
    """A side-step or hold at 15 m must not also start a descent to the pursuit cruise altitude."""
    ctl = make_ctl()
    state = DroneState(connected=True, position=(0.0, 0.0, 15.0), altitude_m=15.0)
    percept = Percept(has_target=True, bearing_deg=30.0, size_deg=20.0, expansion_dps=5.0, altitude_m=15.0)
    without_goal = ctl.motor._clamp(Command(state='AVOID', vy=1.0), percept, state)
    assert without_goal.vz < -1.0, 'with no goal the brain holds its 3 m cruise altitude, so it descends'
    ctl.set_goal(Goal(x=50.0, y=0.0, z=15.0))
    for st in ('AVOID', 'HOLD'):
        cmd = ctl.motor._clamp(Command(state=st, vy=1.0), percept, state)
        assert abs(cmd.vz) < 1e-9, f'descending during {st} with a goal at 15 m: {cmd}'
    cmd = ctl.motor._clamp(Command(state='AVOID'), Percept(altitude_m=13.0), state)
    assert cmd.vz > 0, 'below the goal level it climbs back'


def test_no_odometry_means_hold() -> None:
    ctl = make_ctl()
    ctl.state.connected = False
    ctl.set_goal(Goal(x=10.0, y=0.0, z=15.0))
    cmd = ctl.tick(Percept(has_target=False, altitude_m=15.0))
    assert cmd.state == 'HOLD'
    assert 'odometry' in cmd.reason
    assert cmd.vx == 0.0 and cmd.vy == 0.0


def test_stop_clears_goal_and_arming_calls_bridge() -> None:
    ctl = make_ctl()
    ctl.enable_output(True)
    assert ctl.bridge.armed is True            # NullBridge records the arm call
    ctl.set_goal(Goal(x=10.0, y=0.0, z=15.0))
    assert ctl.goal_status().active
    ctl.stop()
    st = ctl.goal_status()
    assert not st.active and not st.output_enabled


def test_goal_status_geometry() -> None:
    ctl = make_ctl()
    ctl.set_goal(Goal(x=3.0, y=4.0, z=15.0, speed_mps=2.0))
    st = ctl.goal_status()
    assert st.horizontal_m == pytest.approx(5.0)
    assert st.distance_m == pytest.approx(5.0)
    assert st.dz_m == pytest.approx(0.0)
    assert st.eta_s == pytest.approx(2.5)
    assert st.set_at is not None and st.reached_at is None
