"""
Boustrophedon cellular decomposition and the lawnmower path that covers it.

The idea (Choset & Pignon, 1998): sweep a line across the free space. The free part of the
line is a set of intervals; wherever the *number* of intervals changes -- at an obstacle's
left or right extremity, or a reflex vertex of the search polygon -- the space splits or merges,
and that is a cell boundary. Every cell is a region whose sweep-line slice is a single interval
at every x, so it can be covered exactly by back-and-forth ("ox-turning", boustrophedon) lanes.
Cells are then visited greedily nearest-first, with transits routed around the obstacles through
a visibility graph.

Everything is done in a frame rotated so lanes run along +y and the sweep is along x; the
result is rotated back. Lane spacing comes from the camera footprint at the flight height, less
the requested overlap, so coverage is a property of the camera, not a guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from . import geometry as g
from .camera import DownCamera
from .schemas import Cell, CoveragePlan, Point2, Waypoint

Interval = tuple[float, float]


@dataclass
class PlannerConfig:
    altitude_m: float = 15.0
    ground_z_m: float = 0.0
    camera: DownCamera = field(default_factory=DownCamera)
    lane_overlap: float = 0.2
    sweep_heading_deg: float | None = None
    """Direction the lanes run, degrees from +x. None: along the longer side of the area."""
    lane_speed_mps: float = 2.5
    transit_speed_mps: float = 4.0
    lane_arrive_radius_m: float = 1.5
    transit_arrive_radius_m: float = 2.5
    sample_fraction: float = 0.5
    turn_time_s: float = 2.0
    climb_rate_mps: float = 1.5
    start_z_m: float | None = None
    min_lane_spacing_m: float = 1.0
    eps: float = 1e-6


@dataclass
class _Slab:
    x0: float
    x1: float
    left: list[Interval]
    mid: list[Interval]
    right: list[Interval]


@dataclass
class _Cell:
    parts: list[tuple[int, int]]            # (slab index, interval index)
    last_right: Interval


# --- the decomposition ---------------------------------------------------------------


def free_intervals(x: float, area: list[Point2], obstacles: list[list[Point2]]) -> list[Interval]:
    free = g.vertical_slice(area, x)
    blocked = [iv for o in obstacles for iv in g.vertical_slice(o, x)]
    return g.subtract_intervals(free, blocked)


def decompose(
    area: list[Point2], obstacles: list[list[Point2]], eps: float = 1e-6
) -> tuple[list[_Slab], list[_Cell]]:
    """
    Split the free space into boustrophedon cells. ``area`` and ``obstacles`` are in the sweep
    frame (lanes along y). Returns the slabs between consecutive critical x's and the cells,
    each a run of slab-intervals that stayed singly connected.
    """
    xs = sorted({round(p[0], 9) for p in area} | {round(p[0], 9) for o in obstacles for p in o})
    slabs: list[_Slab] = []
    for x0, x1 in zip(xs, xs[1:]):
        if x1 - x0 <= 4 * eps:
            continue
        left = free_intervals(x0 + eps, area, obstacles)
        mid = free_intervals((x0 + x1) / 2.0, area, obstacles)
        right = free_intervals(x1 - eps, area, obstacles)
        if not (len(left) == len(mid) == len(right)):
            # No vertex lies strictly inside a slab, so connectivity is constant across it; a
            # mismatch is numerical (an edge grazing x0+eps). The middle slice is the truth.
            left = right = mid
        slabs.append(_Slab(x0, x1, left, mid, right))

    done: list[_Cell] = []
    open_cells: list[_Cell] = []
    for k, slab in enumerate(slabs):
        n_new = len(slab.left)
        cell_to_new = [
            [i for i in range(n_new) if g.intervals_overlap(c.last_right, slab.left[i])]
            for c in open_cells
        ]
        new_to_cell = [
            [j for j, c in enumerate(open_cells) if g.intervals_overlap(c.last_right, slab.left[i])]
            for i in range(n_new)
        ]
        assigned: list[_Cell | None] = [None] * n_new
        for j, c in enumerate(open_cells):
            targets = cell_to_new[j]
            if len(targets) == 1 and len(new_to_cell[targets[0]]) == 1:
                i = targets[0]                       # one-to-one: the cell continues
                c.parts.append((k, i))
                c.last_right = slab.right[i]
                assigned[i] = c
            else:                                    # split, merge or dead end: it closes
                done.append(c)
        for i in range(n_new):
            if assigned[i] is None:                  # appeared, or born of a split/merge
                assigned[i] = _Cell(parts=[(k, i)], last_right=slab.right[i])
        open_cells = [c for c in assigned if c is not None]
    done.extend(open_cells)
    done.sort(key=lambda c: (slabs[c.parts[0][0]].x0, slabs[c.parts[0][0]].mid[c.parts[0][1]][0]))
    return slabs, done


def cell_polygon(cell: _Cell, slabs: list[_Slab]) -> list[Point2]:
    bottom: list[Point2] = []
    top: list[Point2] = []
    for k, i in cell.parts:
        s = slabs[k]
        bottom += [(s.x0, s.left[i][0]), (s.x1, s.right[i][0])]
        top += [(s.x0, s.left[i][1]), (s.x1, s.right[i][1])]
    pts = bottom + top[::-1]
    out: list[Point2] = []
    for p in pts:
        if not out or g.dist(out[-1], p) > 1e-6:
            out.append(p)
    if len(out) > 1 and g.dist(out[0], out[-1]) <= 1e-6:
        out.pop()
    return out


# --- lanes --------------------------------------------------------------------------


def cell_lanes(
    cell: _Cell,
    slabs: list[_Slab],
    area: list[Point2],
    obstacles: list[list[Point2]],
    spacing: float,
    inset_x: float,
    inset_y: float,
    event_xs: list[float],
) -> list[tuple[float, float, float]]:
    """Lane x positions and y-extents ``(x, y_a, y_b)`` for one cell, sorted by x."""
    x0 = slabs[cell.parts[0][0]].x0
    x1 = slabs[cell.parts[-1][0]].x1
    width = x1 - x0
    if width <= 2.0 * inset_x + 1e-9:
        xs = [(x0 + x1) / 2.0]
    else:
        usable = width - 2.0 * inset_x
        n = int(math.ceil(usable / spacing - 1e-9)) + 1
        xs = [x0 + inset_x + usable * k / (n - 1) for k in range(n)] if n > 1 else [(x0 + x1) / 2.0]
    lanes: list[tuple[float, float, float]] = []
    for x in xs:
        for ex in event_xs:                          # never slice exactly through a vertex
            if abs(x - ex) < 1e-6:
                x = ex + 1e-5 if ex < (x0 + x1) / 2.0 else ex - 1e-5
        part = next(
            ((k, i) for k, i in cell.parts if slabs[k].x0 - 1e-9 <= x <= slabs[k].x1 + 1e-9), None
        )
        if part is None:
            continue
        k, i = part
        ref = slabs[k].mid[i]
        candidates = free_intervals(x, area, obstacles)
        best = max(candidates, key=lambda iv: g.overlap_length(iv, ref), default=None)
        if best is None or g.overlap_length(best, ref) <= 0.0:
            continue
        y0, y1 = best
        if y1 - y0 <= 2.0 * inset_y:
            ya = yb = (y0 + y1) / 2.0
        else:
            ya, yb = y0 + inset_y, y1 - inset_y
        lanes.append((x, ya, yb))
    return lanes


def traversals(lanes: list[tuple[float, float, float]]) -> list[list[tuple[Point2, Point2]]]:
    """The four ways to lawnmower a cell: lanes left-to-right or right-to-left, first lane up or down."""
    out: list[list[tuple[Point2, Point2]]] = []
    for order in (lanes, lanes[::-1]):
        for first_up in (True, False):
            up = first_up
            legs: list[tuple[Point2, Point2]] = []
            for x, ya, yb in order:
                legs.append(((x, ya), (x, yb)) if up else ((x, yb), (x, ya)))
                up = not up
            out.append(legs)
    return out


# --- the plan ------------------------------------------------------------------------


def default_sweep_heading(area: list[Point2]) -> float:
    x0, y0, x1, y1 = g.bounds(area)
    return 0.0 if (x1 - x0) >= (y1 - y0) else 90.0


def plan_coverage(
    area: list[Point2],
    keepouts: list[list[Point2]],
    start_xy: Point2,
    cfg: PlannerConfig,
    keepouts_ignored: list[str] | None = None,
) -> CoveragePlan:
    """Decompose, lay lanes, order cells, route transits, and measure the result."""
    warnings: list[str] = []
    if len(area) < 3 or g.polygon_area(area) < 1.0:
        raise ValueError('search area must be a polygon with non-zero area')
    height = max(cfg.altitude_m - cfg.ground_z_m, 1.0)
    fw, fl = cfg.camera.footprint(height)
    overlap = min(max(cfg.lane_overlap, 0.0), 0.9)
    spacing = max(fw * (1.0 - overlap), cfg.min_lane_spacing_m)
    inset_x = spacing / 2.0
    inset_y = fl * (1.0 - overlap) / 2.0

    heading = cfg.sweep_heading_deg if cfg.sweep_heading_deg is not None else default_sweep_heading(area)
    rot = math.pi / 2.0 - math.radians(heading)      # lanes along +y in the sweep frame
    pivot = g.centroid(area)
    area_r = g.rotate_all(area, rot, pivot)
    obstacles_r = [g.rotate_all(o, rot, pivot) for o in keepouts]
    start_r = g.rotate(start_xy, rot, pivot)

    slabs, cells = decompose(area_r, obstacles_r, eps=cfg.eps)
    event_xs = sorted({s.x0 for s in slabs} | {s.x1 for s in slabs})
    cell_lane_sets = [
        cell_lanes(c, slabs, area_r, obstacles_r, spacing, inset_x, inset_y, event_xs) for c in cells
    ]

    # greedy nearest-first tour over the cells, trying all four traversals of each
    remaining = [i for i, lanes in enumerate(cell_lane_sets) if lanes]
    cur = start_r
    order: list[tuple[int, list[tuple[Point2, Point2]]]] = []
    while remaining:
        best = None
        for ci in remaining:
            for legs in traversals(cell_lane_sets[ci]):
                entry = legs[0][0]
                cost = g.path_length(g.route_around(cur, entry, obstacles_r))
                if best is None or cost < best[0]:
                    best = (cost, ci, legs)
        assert best is not None
        _, ci, legs = best
        order.append((ci, legs))
        remaining.remove(ci)
        cur = legs[-1][1]

    # waypoints, in the sweep frame
    wps_r: list[tuple[Point2, str, int, int]] = []     # (point, kind, cell, lane)
    cur = start_r
    lane_no = 0
    lane_len = 0.0
    for ci, legs in order:
        for a, b in legs:
            route = g.route_around(cur, a, obstacles_r)
            if len(route) == 2 and g.segment_blocked(cur, a, obstacles_r):
                warnings.append(f'transit into cell {ci} lane {lane_no} could not avoid a keep-out')
            for p in route[1:-1]:
                wps_r.append((p, 'transit', -1, -1))
            wps_r.append((a, 'lane_start', ci, lane_no))
            if g.dist(a, b) > 1e-6:
                wps_r.append((b, 'lane_end', ci, lane_no))
                lane_len += g.dist(a, b)
                cur = b
            else:
                cur = a
            lane_no += 1

    # back to the world frame
    waypoints: list[Waypoint] = []
    for p, kind, ci, li in wps_r:
        x, y = g.rotate(p, -rot, pivot)
        is_lane = kind != 'transit'
        waypoints.append(Waypoint(
            x=round(x, 3), y=round(y, 3), z=cfg.altitude_m, kind=kind, cell=ci, lane=li,
            speed_mps=cfg.lane_speed_mps if is_lane else cfg.transit_speed_mps,
            arrive_radius_m=cfg.lane_arrive_radius_m if is_lane else cfg.transit_arrive_radius_m,
        ))
    cells_out = [
        Cell(index=i, polygon=[g.rotate(p, -rot, pivot) for p in cell_polygon(c, slabs)],
             n_lanes=len(cell_lane_sets[i]), area_m2=g.polygon_area(cell_polygon(c, slabs)))
        for i, c in enumerate(cells)
    ]

    pts = [start_xy] + [w.xy() for w in waypoints]
    total = g.path_length(pts)
    transit_len = max(0.0, total - lane_len)
    climb = abs(cfg.altitude_m - cfg.start_z_m) if cfg.start_z_m is not None else 0.0
    est = (lane_len / max(cfg.lane_speed_mps, 0.1) + transit_len / max(cfg.transit_speed_mps, 0.1)
           + lane_no * cfg.turn_time_s + climb / max(cfg.climb_rate_mps, 0.1))
    free_area, covered = coverage_fraction(area, keepouts, waypoints, fw / 2.0, fl / 2.0)
    sample_spacing = max(fl * cfg.sample_fraction, 1.0)
    if not waypoints:
        warnings.append('no free space to cover: every cell is blocked or thinner than a lane')
    return CoveragePlan(
        area=list(area), keepouts=list(keepouts), keepouts_ignored=list(keepouts_ignored or []),
        altitude_m=cfg.altitude_m, height_above_ground_m=height,
        footprint_w_m=fw, footprint_l_m=fl, lane_spacing_m=spacing, lane_overlap=overlap,
        sweep_heading_deg=heading, cells=cells_out, waypoints=waypoints, n_lanes=lane_no,
        lane_length_m=lane_len, transit_length_m=transit_len, path_length_m=total,
        est_duration_s=est, free_area_m2=free_area, coverage_estimate=covered,
        sample_spacing_m=sample_spacing, est_samples=int(math.ceil(total / sample_spacing)),
        warnings=warnings,
    )


def coverage_fraction(
    area: list[Point2],
    keepouts: list[list[Point2]],
    waypoints: list[Waypoint],
    half_w: float,
    half_l: float,
    res: float | None = None,
) -> tuple[float, float]:
    """
    (free area in m^2, fraction of it inside some lane's camera footprint).

    A grid test: a ground point is covered when it is within ``half_w`` across-track of a lane
    segment and within the segment's along-track span extended by ``half_l`` at both ends,
    which is exactly the ground the camera sees while flying that lane. Transits are not
    counted, so this is a lower bound.
    """
    x0, y0, x1, y1 = g.bounds(area)
    res = res or max(0.5, min(half_w, half_l) / 4.0)
    lanes: list[tuple[Point2, Point2]] = []
    for a, b in zip(waypoints, waypoints[1:]):
        if a.kind == 'lane_start' and b.kind == 'lane_end' and a.lane == b.lane:
            lanes.append((a.xy(), b.xy()))
    lanes_with_end = {b.lane for b in waypoints if b.kind == 'lane_end'}
    points: list[Point2] = [w.xy() for w in waypoints if w.kind == 'lane_start' and w.lane not in lanes_with_end]
    n_free = n_cov = 0
    y = y0 + res / 2.0
    while y < y1:
        x = x0 + res / 2.0
        while x < x1:
            p = (x, y)
            if g.point_in_polygon(p, area) and not any(g.point_in_polygon(p, o) for o in keepouts):
                n_free += 1
                hit = False
                for a, b in lanes:
                    along, across = g.point_segment_frame(p, a, b)
                    if abs(across) <= half_w and -half_l <= along <= g.dist(a, b) + half_l:
                        hit = True
                        break
                if not hit:
                    for q in points:
                        if abs(p[0] - q[0]) <= half_w and abs(p[1] - q[1]) <= half_l:
                            hit = True
                            break
                n_cov += hit
            x += res
        y += res
    free_area = n_free * res * res
    return free_area, (n_cov / n_free if n_free else 0.0)
