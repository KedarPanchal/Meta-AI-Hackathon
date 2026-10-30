"""The boustrophedon planner: cells, lanes, coverage, routing, and the camera geometry."""
from __future__ import annotations

import math

import pytest

from mission import geometry as g
from mission.camera import DownCamera
from mission.coverage import PlannerConfig, coverage_fraction, decompose, plan_coverage
from mission.schemas import Area

CAM = DownCamera(hfov_deg=80.0, width=640, height=480)


def cfg(**kw) -> PlannerConfig:
    base = dict(altitude_m=15.0, ground_z_m=0.0, camera=CAM, lane_overlap=0.2)
    base.update(kw)
    return PlannerConfig(**base)


def lane_segments(plan):
    return [(a, b) for a, b in zip(plan.waypoints, plan.waypoints[1:])
            if a.kind == 'lane_start' and b.kind == 'lane_end' and a.lane == b.lane]


def test_box_is_one_cell_with_alternating_lanes() -> None:
    area = Area.box(-20, -20, 20, 20).vertices()
    plan = plan_coverage(area, [], (-25.0, -25.0), cfg())
    assert len(plan.cells) == 1
    fw, fl = CAM.footprint(15.0)
    assert plan.footprint_w_m == pytest.approx(fw) and plan.footprint_l_m == pytest.approx(fl)
    assert plan.lane_spacing_m == pytest.approx(fw * 0.8)
    segs = lane_segments(plan)
    assert plan.n_lanes == len(segs) >= 2
    # lanes are parallel, evenly spaced, and alternate direction
    dirs = []
    for a, b in segs:
        d = (b.x - a.x, b.y - a.y)
        dirs.append(math.copysign(1.0, d[0] if abs(d[0]) > abs(d[1]) else d[1]))
    assert all(x * y < 0 for x, y in zip(dirs, dirs[1:])), dirs
    for w in plan.waypoints:
        assert -20 - 1e-6 <= w.x <= 20 + 1e-6 and -20 - 1e-6 <= w.y <= 20 + 1e-6
        assert w.z == 15.0
    assert plan.coverage_estimate >= 0.99
    assert plan.path_length_m > 0 and plan.est_duration_s > 0
    assert plan.est_samples >= plan.path_length_m / plan.sample_spacing_m - 1


def test_sweep_runs_along_the_long_side_by_default() -> None:
    wide = plan_coverage(Area.box(-60, -10, 60, 10).vertices(), [], (0.0, 0.0), cfg())
    tall = plan_coverage(Area.box(-10, -60, 10, 60).vertices(), [], (0.0, 0.0), cfg())
    assert wide.sweep_heading_deg == 0.0 and tall.sweep_heading_deg == 90.0
    a, b = lane_segments(wide)[0]
    assert abs(b.y - a.y) < 1e-6 and abs(b.x - a.x) > 50
    a, b = lane_segments(tall)[0]
    assert abs(b.x - a.x) < 1e-6 and abs(b.y - a.y) > 50
    rotated = plan_coverage(Area.box(-30, -30, 30, 30).vertices(), [], (0.0, 0.0), cfg(sweep_heading_deg=45.0))
    a, b = lane_segments(rotated)[0]
    ang = math.degrees(math.atan2(b.y - a.y, b.x - a.x)) % 180.0
    assert ang == pytest.approx(45.0, abs=0.5)
    assert rotated.coverage_estimate >= 0.97


def test_obstacle_splits_cells_and_is_never_entered() -> None:
    area = Area.box(-40, -40, 40, 40).vertices()
    obstacle = g.circle_polygon(0.0, 0.0, 8.0, n=12)
    plan = plan_coverage(area, [obstacle], (-45.0, -45.0), cfg(lane_overlap=0.4))
    assert len(plan.cells) >= 3, [c.model_dump() for c in plan.cells]
    for w in plan.waypoints:
        assert not g.point_in_polygon(w.xy(), obstacle), f'waypoint inside the keep-out: {w}'
    for a, b in zip(plan.waypoints, plan.waypoints[1:]):
        assert not g.segment_hits_polygon(a.xy(), b.xy(), obstacle), f'path crosses the keep-out: {a} -> {b}'
    assert plan.coverage_estimate >= 0.97, plan.coverage_estimate
    assert not plan.warnings
    # decomposition detail: the slice through the obstacle has two free intervals
    slabs, cells = decompose(area, [obstacle])
    mid = [s for s in slabs if s.x0 < 1.0 < s.x1][0]       # x=0 is itself a vertex, so an event
    assert len(mid.mid) == 2


def test_transits_route_around_a_wall() -> None:
    """Two lanes separated by a long thin wall: the transit must go around its end."""
    wall = [(-1.0, -30.0), (1.0, -30.0), (1.0, 30.0), (-1.0, 30.0)]
    a, b = (-10.0, 0.0), (10.0, 0.0)
    route = g.route_around(a, b, [wall])
    assert len(route) >= 3
    assert route[0] == a and route[-1] == b
    for p, q in zip(route, route[1:]):
        assert not g.segment_hits_polygon(p, q, wall)
    assert g.path_length(route) < 2 * 31 + 20 + 1.0


def test_polygon_area_and_blocking_obstacle_edge_cases() -> None:
    lshape = [(0, 0), (40, 0), (40, 15), (15, 15), (15, 40), (0, 40)]
    plan = plan_coverage(lshape, [], (0.0, 0.0), cfg())
    assert plan.waypoints
    for w in plan.waypoints:
        assert g.point_in_polygon((w.x + 1e-6, w.y + 1e-6), lshape) or g.point_in_polygon((w.x - 1e-6, w.y - 1e-6), lshape)
    assert plan.coverage_estimate >= 0.95
    # an obstacle covering everything leaves nothing to fly, and the plan says so
    blocked = plan_coverage(Area.box(0, 0, 10, 10).vertices(), [[(-5, -5), (15, -5), (15, 15), (-5, 15)]],
                            (0.0, 0.0), cfg())
    assert blocked.waypoints == []
    assert blocked.warnings
    with pytest.raises(ValueError):
        plan_coverage([(0, 0), (1, 0)], [], (0.0, 0.0), cfg())


def test_lower_altitude_means_more_lanes() -> None:
    area = Area.box(-30, -30, 30, 30).vertices()
    high = plan_coverage(area, [], (0.0, 0.0), cfg(altitude_m=20.0))
    low = plan_coverage(area, [], (0.0, 0.0), cfg(altitude_m=8.0))
    assert low.n_lanes > high.n_lanes
    assert low.footprint_w_m < high.footprint_w_m
    assert low.coverage_estimate >= 0.98 and high.coverage_estimate >= 0.98


def test_coverage_fraction_is_honest() -> None:
    area = Area.box(0, 0, 40, 40).vertices()
    plan = plan_coverage(area, [], (0.0, 0.0), cfg())
    half = plan.waypoints[: len(plan.waypoints) // 2]
    full_area, full = coverage_fraction(area, [], plan.waypoints, plan.footprint_w_m / 2, plan.footprint_l_m / 2)
    _, partial = coverage_fraction(area, [], half, plan.footprint_w_m / 2, plan.footprint_l_m / 2)
    assert full_area == pytest.approx(1600.0, rel=0.05)
    assert full > partial


def test_camera_round_trip_and_footprint() -> None:
    cam = CAM
    fw, fl = cam.footprint(15.0)
    assert fw == pytest.approx(2 * 15 * math.tan(math.radians(40)))
    assert fl < fw
    drone = (5.0, -3.0, 15.0)
    for yaw in (0.0, 0.7, -2.1):
        for target in ((7.0, -1.0), (0.0, -9.0), (5.0, -3.0)):
            uv = cam.ground_to_pixel(target, drone, yaw, 0.0)
            assert uv is not None
            back = cam.pixel_to_ground(uv[0], uv[1], drone, yaw, 0.0)
            assert back[0] == pytest.approx(target[0], abs=1e-6)
            assert back[1] == pytest.approx(target[1], abs=1e-6)
    # image top is forward: a point ahead of a drone facing east lands in the top half
    ahead = cam.ground_to_pixel((12.0, -3.0), drone, 0.0, 0.0)     # 7 m ahead, inside the 9.4 m half-length
    assert ahead is not None and ahead[1] < 0.5 and abs(ahead[0] - 0.5) < 1e-6
    # and a point to its right lands in the right half
    right = cam.ground_to_pixel((5.0, -8.0), drone, 0.0, 0.0)
    assert right is not None and right[0] > 0.5
    assert cam.ground_to_pixel((500.0, 0.0), drone, 0.0, 0.0) is None


def test_geometry_helpers() -> None:
    sq = [(0, 0), (4, 0), (4, 4), (0, 4)]
    assert g.polygon_area(sq) == 16.0
    assert g.point_in_polygon((2, 2), sq) and not g.point_in_polygon((5, 2), sq)
    assert g.vertical_slice(sq, 2.0) == [(0.0, 4.0)]
    assert g.subtract_intervals([(0.0, 10.0)], [(3.0, 4.0), (8.0, 12.0)]) == [(0.0, 3.0), (4.0, 8.0)]
    grown = g.buffer_polygon(sq, 1.0)
    assert g.polygon_area(grown) == pytest.approx(36.0)
    circ = g.circle_polygon(0, 0, 2.0, n=8)
    assert all(math.hypot(x, y) >= 2.0 - 1e-9 for x, y in circ)
    assert g.segments_cross((0, 0), (2, 2), (0, 2), (2, 0))
    assert not g.segments_cross((0, 0), (1, 1), (2, 2), (3, 3))
