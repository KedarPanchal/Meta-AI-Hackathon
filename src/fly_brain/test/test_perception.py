"""The loom tracker: per-object expansion, urgency ranking, and two sensors sharing it."""
from __future__ import annotations

import math

from fly_brain.perception import LoomTracker, Threat, bearing_of, size_of, threat_from_box
from fly_brain.schemas import Box


def box(size_px: float, cx: float = 320.0, cy: float = 240.0, tid: int = 1) -> Box:
    return Box(x=cx - size_px / 2, y=cy - size_px / 2, w=size_px, h=size_px, track_id=tid)


def test_box_path_matches_threat_path() -> None:
    a, b = LoomTracker(hfov_deg=60.0), LoomTracker(hfov_deg=60.0)
    for i, s in enumerate((20.0, 24.0, 30.0)):
        pa = a.update([box(s)], t=i * 0.1)
        pb = b.update_threats([threat_from_box(box(s), 60.0)], t=i * 0.1)
        assert pa == pb
    assert pa.has_target and pa.size_deg > 0 and pa.expansion_dps > 0
    assert pa.distance_m is None                     # a camera measures no range


def test_bearing_and_size_from_pinhole() -> None:
    b = box(64.0, cx=640.0, cy=240.0)                # right edge of a 60 degree camera
    bearing, elevation = bearing_of(b, 60.0)
    assert math.isclose(bearing, 30.0, abs_tol=0.01) and abs(elevation) < 1e-9
    full_width = Box(x=0.0, y=200.0, w=640.0, h=80.0)
    assert math.isclose(size_of(full_width, 60.0), 60.0, abs_tol=0.01)


def test_expansion_is_per_object_not_per_frame() -> None:
    """Two objects of different size seen together must not produce spurious expansion."""
    tr = LoomTracker(hfov_deg=60.0)
    small, big = box(20.0, tid=1), box(80.0, cx=100.0, tid=2)
    for i in range(5):
        p = tr.update([small, big], t=i * 0.1)
    assert p.expansion_dps == 0.0
    assert p.size_deg == size_of(big, 60.0)          # nothing grows, so the biggest wins


def test_smallest_time_to_contact_outranks_biggest_size() -> None:
    tr = LoomTracker(hfov_deg=60.0)
    for i, s in enumerate((20.0, 26.0, 34.0, 44.0)):
        p = tr.update([box(s, tid=1), box(120.0, cx=100.0, tid=2)], t=i * 0.1)
    assert tr.last is not None and tr.last.track_id == 1
    assert p.size_deg == size_of(box(44.0), 60.0) and p.expansion_dps > 0


def test_sources_do_not_collide_and_lidar_range_passes_through() -> None:
    tr = LoomTracker(hfov_deg=60.0)
    cam = threat_from_box(box(20.0, tid=1), 60.0)
    lid = Threat(bearing_deg=10.0, elevation_deg=0.0, size_deg=5.0, track_id=1,
                 source='lidar', distance_m=6.0)
    tr.update_threats([cam, lid], t=0.0)
    lid1 = Threat(bearing_deg=10.0, elevation_deg=0.0, size_deg=7.0, track_id=1,
                  source='lidar', distance_m=4.0)
    tr.update_threats([cam, lid1], t=0.1)
    lid2 = Threat(bearing_deg=10.0, elevation_deg=0.0, size_deg=11.0, track_id=1,
                  source='lidar', distance_m=3.0)
    p = tr.update_threats([cam, lid2], t=0.2)
    assert ('camera', 1) in tr.history and ('lidar', 1) in tr.history
    assert tr.last is not None and tr.last.source == 'lidar'
    assert p.distance_m == 3.0 and p.bearing_deg == 10.0
    assert math.isclose(p.expansion_dps, 0.7 * 40.0)  # third sample: smoothed first derivative


def test_stationary_threat_passes_moving_flag_and_never_looms() -> None:
    """A lidar-static wall reaches the percept as moving=False with zero expansion,
    so the motor can analyse an over/under/sideways pass instead of an escape."""
    tr = LoomTracker(hfov_deg=60.0)
    p = None
    for i, s in enumerate((10.0, 16.0, 24.0)):
        wall = Threat(bearing_deg=5.0, elevation_deg=0.0, size_deg=s, track_id=3,
                      source='lidar', distance_m=3.0, moving=False)
        p = tr.update_threats([wall], t=i * 0.1)
    assert p is not None and p.moving is False
    assert p.expansion_dps == 0.0 and p.distance_m == 3.0


def test_repeated_sample_keeps_the_estimate() -> None:
    """A camera frame that has not changed since the last tick is not zero expansion."""
    tr = LoomTracker(hfov_deg=60.0)
    tr.update_threats([threat_from_box(box(20.0), 60.0, t=0.0)], t=0.0)
    tr.update_threats([threat_from_box(box(25.0), 60.0, t=0.05)], t=0.05)
    p1 = tr.update_threats([threat_from_box(box(30.0), 60.0, t=0.1)], t=0.1)
    p2 = tr.update_threats([threat_from_box(box(30.0), 60.0, t=0.1)], t=0.15)   # same frame
    assert p1.expansion_dps > 0 and p2.expansion_dps == p1.expansion_dps
    p3 = tr.update_threats([threat_from_box(box(36.0), 60.0, t=0.2)], t=0.2)
    assert p3.expansion_dps > 0


def test_history_is_pruned_when_objects_vanish() -> None:
    tr = LoomTracker(hfov_deg=60.0)
    tr.update([box(20.0, tid=1), box(20.0, cx=100.0, tid=2)], t=0.0)
    tr.update([box(22.0, tid=1)], t=0.1)
    assert set(tr.history) == {('camera', 1)}
    p = tr.update([], t=0.2)
    assert not p.has_target and not tr.history and tr.last is None


def test_two_samples_are_not_enough_to_loom() -> None:
    tr = LoomTracker(hfov_deg=60.0)
    tr.update([box(20.0)], t=0.0)
    p = tr.update([box(60.0)], t=0.1)                  # a merge, or a new id: a leap in size
    assert p.expansion_dps == 0.0
    p = tr.update([box(72.0)], t=0.2)                  # third sample: now it may loom
    assert p.expansion_dps > 0.0


def test_slow_growth_of_a_big_thing_is_scenery() -> None:
    tr = LoomTracker(hfov_deg=60.0)
    # a 30 degree object growing 3 degrees per 0.1 s: tau about 1.5 s, it looms...
    for i, s in enumerate((300.0, 330.0, 360.0, 390.0)):
        p = tr.update([Box(x=0.0, y=0.0, w=s, h=100.0)], t=i * 0.1)
    assert p.expansion_dps > 0.0 and p.size_deg / p.expansion_dps < 3.0
    # ...but at 0.3 degrees per 0.1 s (tau about 15 s) it is just a tree getting a little closer
    tr = LoomTracker(hfov_deg=60.0)
    for i, s in enumerate((300.0, 301.5, 303.0, 304.5, 306.0)):
        p = tr.update([Box(x=0.0, y=0.0, w=s, h=100.0)], t=i * 0.1)
    assert p.has_target and p.expansion_dps == 0.0
