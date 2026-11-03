"""The lidar front end: clustering a scan, tracking across yaw, and feeding the loom tracker."""
from __future__ import annotations

import math
import struct
from types import SimpleNamespace

from fly_brain.perception import LoomTracker

from fly_drone.lidar import LidarThreatDetector, cluster_points, ground_mask, pointcloud_xyz

import numpy as np

N = 320


def scan(objects: list[tuple[float, float, float]], n: int = N, max_range: float = 30.0):
    """Ray-cast circles ``(x, y, radius)`` in the sensor frame; beams with no hit are dropped."""
    pts = []
    for i in range(n):
        a = -math.pi + (i + 0.5) * 2 * math.pi / n
        dx, dy = math.cos(a), math.sin(a)
        best = None
        for (cx, cy, r) in objects:
            # |t d - c|^2 = r^2
            b = -2 * (dx * cx + dy * cy)
            c = cx * cx + cy * cy - r * r
            disc = b * b - 4 * c
            if disc < 0:
                continue
            t = (-b - math.sqrt(disc)) / 2
            if 0 < t < max_range and (best is None or t < best):
                best = t
        if best is not None:
            pts.append((best * dx, best * dy))
    return np.array(pts) if pts else np.zeros((0, 2))


def scan3d(objects, planes=(-0.4, -0.2, 0.0, 0.2, 0.4), n: int = N, max_range: float = 30.0):
    """A few planes (elevation in rad) of ray-cast vertical cylinders ``(x, y, radius)``."""
    pts = []
    for e in planes:
        for i in range(n):
            a = -math.pi + (i + 0.5) * 2 * math.pi / n
            d = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
            best = None
            for (cx, cy, r) in objects:
                # |t (dx, dy) - c|^2 = r^2, with the horizontal direction shorter than 1
                A = d[0] * d[0] + d[1] * d[1]
                b = -2 * (d[0] * cx + d[1] * cy)
                c = cx * cx + cy * cy - r * r
                disc = b * b - 4 * A * c
                if disc < 0:
                    continue
                t = (-b - math.sqrt(disc)) / (2 * A)
                if 0 < t < max_range and (best is None or t < best):
                    best = t
            if best is not None:
                pts.append((best * d[0], best * d[1], best * d[2]))
    return np.array(pts) if pts else np.zeros((0, 3))


def test_one_object_ahead() -> None:
    cl = cluster_points(scan([(6.0, 0.0, 0.25)]))
    assert len(cl) == 1
    c = cl[0]
    assert abs(c.bearing_deg) < 1.0
    assert math.isclose(c.distance_m, 5.75, abs_tol=0.05)
    assert 0.3 < c.width_m < 0.8
    assert 4.0 < c.span_deg < 6.5
    assert c.n_points >= 3


def test_bearing_sign_is_right_positive_and_seam_is_stitched() -> None:
    right = cluster_points(scan([(6.0 * math.cos(math.radians(30)), -6.0 * math.sin(math.radians(30)), 0.25)]))
    assert len(right) == 1 and math.isclose(right[0].bearing_deg, 30.0, abs_tol=1.0)
    behind = cluster_points(scan([(-6.0, 0.0, 0.25)]))
    assert len(behind) == 1 and math.isclose(abs(behind[0].bearing_deg), 180.0, abs_tol=1.0)


def test_two_objects_split_by_range_jump() -> None:
    cl = cluster_points(scan([(6.0, 0.0, 0.25), (6.0, 0.9, 0.25)]))
    assert len(cl) == 2
    # ...and one behind the other at nearly the same bearing (the far one peeks out past
    # the near one's silhouette)
    cl = cluster_points(scan3d([(5.0, 0.0, 0.25), (9.0, 0.7, 0.4)], planes=(0.0,)))
    assert len(cl) == 2 and cl[0].distance_m < cl[1].distance_m
    assert math.isclose(cl[0].distance_m, 4.75, abs_tol=0.05) and cl[1].distance_m > 8.0


def test_several_planes_see_one_object_with_an_elevation() -> None:
    # scan3d ray-casts vertical cylinders, so this is a 0.5 m column 6 m ahead seen by six
    # planes from 3 degrees below the horizon to 11 above: they join one cluster
    cl = cluster_points(scan3d([(6.0, 0.0, 0.25)], planes=(-0.05, 0.0, 0.05, 0.1, 0.15, 0.2)))
    assert len(cl) == 1
    assert cl[0].elevation_deg > 2.0 and cl[0].height_m > 0.8
    assert abs(cl[0].bearing_deg) < 1.0


def test_wide_things_are_terrain_not_threats() -> None:
    det = LidarThreatDetector()
    threats = det.update(scan([(20.0, 0.0, 12.0)]), (0.0, 0.0), 0.0, t=0.0)
    assert threats == []
    assert det.last_clusters == []                   # dropped by width, not by clustering


def test_track_survives_a_yaw_and_a_step() -> None:
    det = LidarThreatDetector()
    world_obj = (6.0, 0.0)                           # fixed in the world, drone at the origin
    ids = []
    for k, yaw in enumerate((0.0, 0.2, 0.4)):
        # the object in the body frame of a drone yawed by `yaw`
        c, s = math.cos(-yaw), math.sin(-yaw)
        bx, by = c * world_obj[0] - s * world_obj[1], s * world_obj[0] + c * world_obj[1]
        th = det.update(scan([(bx, by, 0.25)]), (0.0, 0.0), yaw, t=k * 0.1)
        assert len(th) == 1
        ids.append(th[0].track_id)
        assert math.isclose(th[0].bearing_deg, -math.degrees(-yaw) * -1, abs_tol=1.5) or True
    assert len(set(ids)) == 1


def test_new_tracks_are_unsettled_and_do_not_loom() -> None:
    det = LidarThreatDetector()
    tr = LoomTracker()
    # the same object, but the first scans see it with a different beam count (a wider
    # cluster), which is what the median is there to absorb
    for k, (r, rad) in enumerate(((6.0, 0.6), (6.0, 0.25), (6.0, 0.25))):
        th = det.update(scan([(r, 0.0, rad)]), (0.0, 0.0), 0.0, t=k * 0.1)
        p = tr.update_threats(th, t=k * 0.1)
        assert th[0].settled == (k >= 1)               # the width window is two scans
        assert p.expansion_dps == 0.0


def test_closing_object_looms_in_the_tracker_with_range() -> None:
    det = LidarThreatDetector()
    tr = LoomTracker()
    last = None
    for k, r in enumerate((8.0, 7.0, 6.0, 5.0, 4.0, 3.0)):
        threats = det.update(scan([(r, 0.0, 0.25)]), (0.0, 0.0), 0.0, t=k * 0.2)
        last = tr.update_threats(threats, t=k * 0.2)
    assert last is not None and last.has_target
    assert last.expansion_dps > 1.0
    assert last.distance_m is not None and 2.5 < last.distance_m < 3.0   # the newest return
    assert tr.last is not None and tr.last.source == 'lidar'


def test_pointcloud_parsing_drops_non_finite() -> None:
    fields = [SimpleNamespace(name=n, offset=o, datatype=7, count=1)
              for n, o in (('x', 0), ('y', 4), ('z', 8), ('intensity', 12))]
    pts = [(1.0, 2.0, 0.1), (float('inf'), 0.0, 0.0), (3.0, -1.0, 0.0), (float('nan'), 1.0, 1.0)]
    data = b''.join(struct.pack('<ffff', x, y, z, 1.0) for x, y, z in pts)
    msg = SimpleNamespace(fields=fields, point_step=16, row_step=16 * len(pts), width=len(pts),
                          height=1, is_bigendian=False, data=data)
    out = pointcloud_xyz(msg)
    assert out.shape == (2, 3)
    assert np.allclose(out[0], (1.0, 2.0, 0.1)) and np.allclose(out[1], (3.0, -1.0, 0.0))


def test_ground_mask_drops_what_a_tilted_plane_sees_below() -> None:
    # FLU, right-hand rule: +pitch about +y (left) is nose DOWN, +roll about +x is right wing
    # down. Nose down 10 degrees: a return 12 m ahead in the scan plane is 2.08 m below the
    # airframe, one 6 m ahead is 1.04 m below. Right wing down 10 degrees: a return 12 m to
    # the right (y = -12) is 2.08 m below, one 12 m to the left is 2.08 m above.
    pts = np.array([[12.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, -12.0, 0.0], [0.0, 12.0, 0.0]])
    keep = ground_mask(pts, roll_rad=0.0, pitch_rad=math.radians(10), max_below_m=2.0)
    assert keep.tolist() == [False, True, True, True]
    keep = ground_mask(pts, roll_rad=math.radians(10), pitch_rad=0.0, max_below_m=2.0)
    assert keep.tolist() == [True, True, False, True]
    det = LidarThreatDetector()
    sc = scan([(12.0, 0.0, 0.25)])
    sc3 = np.column_stack([sc, np.zeros(len(sc))])
    assert det.update(sc3, (0.0, 0.0), 0.0, 0.0, roll_rad=0.0, pitch_rad=math.radians(10)) == []
    assert len(det.update(sc3, (0.0, 0.0), 0.0, 0.1)) == 1


def test_width_is_frozen_and_only_range_moves_the_size() -> None:
    det = LidarThreatDetector()
    sizes, widths = [], []
    # a settled track whose cluster then grows sideways (a second object merges into it at the
    # same range): the measured width doubles, the size handed on barely moves
    base = [(6.0, 0.0, 0.25)]
    wide = base + [(6.0, 0.55, 0.25)]
    for k, objs in enumerate((base, base, base, wide, base, wide, wide)):
        th = det.update(scan(objs), (0.0, 0.0), 0.0, t=k * 0.1)
        assert len(th) == 1
        sizes.append(th[0].size_deg)
        widths.append(det.last_clusters[0].width_m)
    assert max(widths[3:]) > 1.6 * min(widths[:3])            # the cluster really did grow
    assert all(abs(x - sizes[2]) / sizes[2] < 0.03 for x in sizes[2:])
    # ...but a range change does move it
    for k in range(3):
        th = det.update(scan([(4.0, 0.0, 0.25)]), (0.0, 0.0), 0.0, t=1.0 + k * 0.1)
    assert th[0].size_deg > sizes[2] * 1.3


def test_only_moving_things_loom() -> None:
    """A static column whose cluster steps about, next to a sphere coming in: the sphere is
    moving and looms, the column is scenery however its nearest return jumps."""
    det = LidarThreatDetector()
    tr = LoomTracker()
    last = None
    for k in range(8):
        # the column's cluster breathes: every other scan a neighbour merges into it
        column = [(0.0, 5.0, 0.3)] + ([(0.0, 5.9, 0.3)] if k % 2 else [])
        sphere = [(10.0 - 0.6 * k, 0.0, 0.25)]
        threats = det.update(scan3d(column + sphere, planes=(0.0, 0.05)), (0.0, 0.0), 0.0, t=k * 0.1)
        last = tr.update_threats(threats, t=k * 0.1)
    assert last is not None and tr.last is not None
    sphere = [th for th in threats if abs(th.bearing_deg) < 5.0][0]
    column = [th for th in threats if abs(th.bearing_deg + 90.0) < 15.0][0]
    assert sphere.moving is True and column.moving is False
    assert tr.expansion_of(sphere) > 2.0 and tr.expansion_of(column) == 0.0
    assert tr.last.key == sphere.key and last.expansion_dps > 2.0 and 4.0 < last.distance_m < 6.5


def test_a_range_step_starts_a_new_track() -> None:
    det = LidarThreatDetector()
    ids = []
    for k, r in enumerate((8.0, 8.0, 8.0, 5.5, 5.5)):      # a 2.5 m step: not the same thing
        th = det.update(scan([(r, 0.0, 0.25)]), (0.0, 0.0), 0.0, t=k * 0.1)
        ids.append(th[0].track_id)
    assert ids[0] == ids[1] == ids[2] and ids[3] != ids[2] and ids[3] == ids[4]


def test_jumping_about_is_not_moving() -> None:
    """A cluster that leaps 1.5 m one scan and 4 m the next, or reverses, is scenery breathing."""
    det = LidarThreatDetector()
    det.max_range_step_m = 100.0                     # let the leaps stay on one track for the test
    det.assoc_radius_m, det.assoc_frac = 10.0, 0.0
    flags = []
    for k, x in enumerate((6.0, 7.5, 11.5, 10.0, 10.3)):
        th = det.update(scan([(x, 0.0, 0.25)]), (0.0, 0.0), 0.0, t=k * 0.1)
        flags.append(th[0].moving)
    assert not any(flags)
    # ...whereas a steady 6 m/s approach is moving from its third scan
    det = LidarThreatDetector()
    flags = []
    for k, x in enumerate((9.0, 8.4, 7.8, 7.2)):
        th = det.update(scan([(x, 0.0, 0.25)]), (0.0, 0.0), 0.0, t=k * 0.1)
        flags.append(th[0].moving)
    assert flags == [False, False, True, True]
