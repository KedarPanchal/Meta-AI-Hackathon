"""End to end on the in-process sim: the real planner, runner and brain navigation, a kinematic
world, a synthetic downward camera and the colour detector. The planner is never told where
the person is."""
from __future__ import annotations

import math
import time

import pytest

from fly_brain.controller import reset_controller

from mission.config import reload_mission_settings
from mission.runner import MissionRequest, reset_service, service
from mission.schemas import Area, Keepout

PERSON = (2.0, -38.0)        # person_target in forest_search.world; only the sim camera knows


@pytest.fixture()
def svc():
    reset_service()
    reset_controller()
    reload_mission_settings()
    s = service()
    yield s
    reset_service()
    reset_controller()


def request(mode: str, area: Area, **kw) -> MissionRequest:
    cfg = service().cfg
    base = dict(
        mode=mode, area=area, altitude_m=15.0, speed_mps=cfg.speed_mps,
        transit_speed_mps=cfg.transit_speed_mps, lane_overlap=0.2, keepouts=[],
        sweep_heading_deg=None, min_confidence=0.6, confirm=True, return_home=False,
    )
    base.update(kw)
    return MissionRequest(**base)


def wait_for(mission, states: set[str], timeout_s: float):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if mission.state in states:
            return
        time.sleep(0.05)
    raise AssertionError(f'mission still {mission.state} after {timeout_s}s: {list(mission._log)[-8:]}')


def test_search_finds_the_person_and_hovers_over_them(svc) -> None:
    area = Area.box(-20, -56, 24, -20)
    m = svc.start(request('search', area))
    assert m.plan.coverage_estimate >= 0.97
    wait_for(m, {'found', 'complete', 'failed', 'aborted'}, 90.0)
    assert m.state == 'found', (m.state, m.note, list(m._log))
    best = m.best
    assert best is not None and best.confirmed
    assert best.estimated_xy is not None
    err = math.hypot(best.estimated_xy[0] - PERSON[0], best.estimated_xy[1] - PERSON[1])
    assert err < 3.0, f'estimate off by {err:.1f} m: {best.estimated_xy}'
    # the drone ends up holding above them, armed, and the brain is still navigating
    time.sleep(1.0)
    st = svc.status()
    assert st.state == 'found' and st.progress == 1.0
    assert st.brain is not None and st.brain.active and st.brain.output_enabled
    assert st.brain.goal is not None and st.brain.goal.label == 'over person'
    assert st.position is not None
    assert math.hypot(st.position[0] - PERSON[0], st.position[1] - PERSON[1]) < 4.0
    assert st.position[2] == pytest.approx(15.0, abs=1.5)
    assert any('FOUND' in line for line in st.log)
    assert svc.link.goals_sent >= 2


def test_scout_covers_everything_and_keeps_sightings(svc) -> None:
    area = Area.box(-14, -52, 18, -24)
    m = svc.start(request('scout', area, confirm=False))
    n = len(m.plan.waypoints)
    wait_for(m, {'complete', 'failed', 'aborted', 'found'}, 120.0)
    assert m.state == 'complete', (m.state, m.note, list(m._log))
    assert m.waypoint_index == n
    assert m.timeouts == 0, list(m._log)
    assert len(m.sightings) >= 1
    best = m.best
    assert best is not None and best.estimated_xy is not None
    assert math.hypot(best.estimated_xy[0] - PERSON[0], best.estimated_xy[1] - PERSON[1]) < 3.0
    # nearby positives were merged into one sighting rather than listed over and over
    assert len(m.sightings) <= 2
    st = svc.status()
    assert st.progress == 1.0
    assert st.brain is not None and st.brain.goal is not None and 'hold' in st.brain.goal.label


def test_abort_holds_position(svc) -> None:
    m = svc.start(request('search', Area.box(-58, -20, 58, 58)))  # the empty northern two-thirds
    time.sleep(1.0)
    assert m.state == 'running'
    st = svc.abort()
    assert st.state == 'aborted'
    assert st.brain is not None and st.brain.goal is not None
    assert 'abort' in st.brain.goal.label
    pose = svc.link.pose()
    assert math.hypot(st.brain.goal.x - pose.x, st.brain.goal.y - pose.y) < 3.0
    assert st.brain.goal.z == pytest.approx(15.0)
    # a second mission may start right away
    m2 = svc.start(request('scout', Area.box(-10, -10, 10, 10)))
    assert m2.state in ('planned', 'running')
    svc.abort()


def test_one_mission_at_a_time(svc) -> None:
    svc.start(request('search', Area.box(-58, -20, 58, 58)))
    with pytest.raises(RuntimeError, match='abort_mission'):
        svc.start(request('search', Area.box(-5, -5, 5, 5)))
    svc.abort()


def test_keepouts_only_matter_at_their_height(svc) -> None:
    tower = Keepout(name='tower', x=0.0, y=0.0, radius_m=3.0, top_z_m=10.5)
    high = svc.plan(request('scout', Area.box(-30, -30, 30, 30), altitude_m=15.0, keepouts=[tower]), None)
    low = svc.plan(request('scout', Area.box(-30, -30, 30, 30), altitude_m=8.0, keepouts=[tower]), None)
    assert high.keepouts == [] and high.keepouts_ignored == ['tower']
    assert len(low.keepouts) == 1 and low.keepouts_ignored == []
    assert len(low.cells) >= 3
    defaults = svc.default_keepouts()
    assert {k.name for k in defaults} >= {'lookout_tower', 'signal_mast', 'radio_pole'}


def test_look_once_reports_what_the_camera_sees(svc) -> None:
    link = svc.link
    out = svc.look()
    assert 'detection' in out and out['detection']['person_visible'] is False
    # teleport the sim drone over the person and look again
    link.x, link.y, link.z, link.yaw = PERSON[0] + 1.0, PERSON[1] - 1.0, 15.0, 0.4
    out = svc.look()
    det = out['detection']
    assert det['person_visible'] is True and det['confidence'] >= 0.6
    assert out['estimated_xy'] is not None
    assert math.hypot(out['estimated_xy'][0] - PERSON[0], out['estimated_xy'][1] - PERSON[1]) < 1.5
    assert out['detector'].startswith('color')


def test_status_before_any_mission_is_informative(svc) -> None:
    st = svc.status()
    assert st.state == 'idle'
    assert 'search_for_person' in st.note
    assert st.id is None
