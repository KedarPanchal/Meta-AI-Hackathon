"""
Flat 2D geometry for the planner. Pure Python, no shapely: the sim environment does not have it
and nothing here needs more than a few hundred vertices.

Conventions: points are ``(x, y)`` tuples, polygons are vertex lists in either winding, and the
interior test is even-odd so self-touching polygons still behave.
"""
from __future__ import annotations

import heapq
import math
from typing import Iterable, Sequence

Point = tuple[float, float]
Interval = tuple[float, float]
Polygon = Sequence[Point]

EPS = 1e-9


def rotate(p: Point, angle_rad: float, about: Point = (0.0, 0.0)) -> Point:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y = p[0] - about[0], p[1] - about[1]
    return (about[0] + c * x - s * y, about[1] + s * x + c * y)


def rotate_all(pts: Iterable[Point], angle_rad: float, about: Point = (0.0, 0.0)) -> list[Point]:
    return [rotate(p, angle_rad, about) for p in pts]


def dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def path_length(pts: Sequence[Point]) -> float:
    return sum(dist(a, b) for a, b in zip(pts, pts[1:]))


def polygon_area(poly: Polygon) -> float:
    """Unsigned shoelace area."""
    s = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def centroid(poly: Polygon) -> Point:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def bounds(pts: Iterable[Point]) -> tuple[float, float, float, float]:
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)


def point_in_polygon(p: Point, poly: Polygon) -> bool:
    """Even-odd rule. Points exactly on an edge may go either way; callers nudge."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xi:
                inside = not inside
    return inside


def vertical_slice(poly: Polygon, x: float) -> list[Interval]:
    """
    The y-intervals where the vertical line at ``x`` is inside the polygon.

    Each edge that straddles ``x`` contributes one crossing; sorted crossings pair up into
    inside intervals (even-odd). A vertex exactly at ``x`` is handled by the half-open rule, so
    callers that slice at vertex x-coordinates should offset by an epsilon.
    """
    ys: list[float] = []
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (x0 > x) != (x1 > x):
            ys.append(y0 + (x - x0) * (y1 - y0) / (x1 - x0))
    ys.sort()
    return [(ys[i], ys[i + 1]) for i in range(0, len(ys) - 1, 2)]


def subtract_intervals(free: list[Interval], blocked: list[Interval]) -> list[Interval]:
    """Remove every blocked interval from every free one. Both lists may be unsorted."""
    out = list(free)
    for b0, b1 in blocked:
        nxt: list[Interval] = []
        for f0, f1 in out:
            if b1 <= f0 or b0 >= f1:
                nxt.append((f0, f1))
                continue
            if b0 > f0:
                nxt.append((f0, b0))
            if b1 < f1:
                nxt.append((b1, f1))
        out = nxt
    return sorted((a, b) for a, b in out if b - a > EPS)


def intervals_overlap(a: Interval, b: Interval, eps: float = EPS) -> bool:
    return a[0] < b[1] - eps and b[0] < a[1] - eps


def overlap_length(a: Interval, b: Interval) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def circle_polygon(cx: float, cy: float, r: float, n: int = 12) -> list[Point]:
    """A regular polygon that *contains* the circle (circumscribed), so clearance is not lost."""
    rr = r / math.cos(math.pi / n)
    return [
        (cx + rr * math.cos(2 * math.pi * k / n), cy + rr * math.sin(2 * math.pi * k / n))
        for k in range(n)
    ]


def buffer_polygon(poly: Polygon, d: float) -> list[Point]:
    """
    Grow a polygon outward by ``d`` by pushing each vertex along its angle bisector.

    Exact for convex polygons; for concave ones the reflex vertices move inward along their
    bisector, which is still a superset of the original for the small buffers used here.
    """
    if d <= 0:
        return list(poly)
    n = len(poly)
    area2 = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        area2 += x0 * y1 - x1 * y0
    ccw = area2 > 0
    out: list[Point] = []
    for i in range(n):
        p_prev, p, p_next = poly[i - 1], poly[i], poly[(i + 1) % n]
        e0 = (p[0] - p_prev[0], p[1] - p_prev[1])
        e1 = (p_next[0] - p[0], p_next[1] - p[1])
        l0, l1 = math.hypot(*e0) or 1.0, math.hypot(*e1) or 1.0
        # outward normals of the two edges meeting at p
        if ccw:
            n0, n1 = (e0[1] / l0, -e0[0] / l0), (e1[1] / l1, -e1[0] / l1)
        else:
            n0, n1 = (-e0[1] / l0, e0[0] / l0), (-e1[1] / l1, e1[0] / l1)
        bx, by = n0[0] + n1[0], n0[1] + n1[1]
        bl = math.hypot(bx, by)
        if bl < 1e-9:                      # straight-through vertex
            out.append((p[0] + n0[0] * d, p[1] + n0[1] * d))
            continue
        bx, by = bx / bl, by / bl
        cos_half = max(0.2, bx * n0[0] + by * n0[1])   # cap the miter at very sharp corners
        k = d / cos_half
        out.append((p[0] + bx * k, p[1] + by * k))
    return out


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_cross(a: Point, b: Point, c: Point, d: Point, eps: float = 1e-9) -> bool:
    """Proper crossing of open segments ab and cd (touching at an endpoint does not count)."""
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def segment_hits_polygon(a: Point, b: Point, poly: Polygon) -> bool:
    """Does the segment cross into the polygon? Edge crossings or a midpoint strictly inside."""
    n = len(poly)
    for i in range(n):
        if segments_cross(a, b, poly[i], poly[(i + 1) % n]):
            return True
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return point_in_polygon(mid, poly)


def segment_blocked(a: Point, b: Point, obstacles: Sequence[Polygon]) -> bool:
    return any(segment_hits_polygon(a, b, o) for o in obstacles)


def route_around(a: Point, b: Point, obstacles: Sequence[Polygon]) -> list[Point]:
    """
    Shortest obstacle-free polyline from ``a`` to ``b`` through the obstacles' vertices.

    A visibility graph over the (already buffered) obstacle vertices, each pushed a hair outward
    so a route along a face is not counted as crossing it. Falls back to the straight line, with
    the caller warned by ``segment_blocked``, if no route exists at all.
    """
    if not obstacles or not segment_blocked(a, b, obstacles):
        return [a, b]
    nodes: list[Point] = [a, b]
    for poly in obstacles:
        cx, cy = centroid(poly)
        for x, y in poly:
            dx, dy = x - cx, y - cy
            l = math.hypot(dx, dy) or 1.0
            nodes.append((x + dx / l * 0.05, y + dy / l * 0.05))
    n = len(nodes)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not segment_blocked(nodes[i], nodes[j], obstacles):
                w = dist(nodes[i], nodes[j])
                adj[i].append((j, w))
                adj[j].append((i, w))
    best = [math.inf] * n
    prev = [-1] * n
    best[0] = 0.0
    heap = [(0.0, 0)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > best[u]:
            continue
        if u == 1:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < best[v]:
                best[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if best[1] == math.inf:
        return [a, b]
    path = []
    u = 1
    while u != -1:
        path.append(nodes[u])
        u = prev[u]
    return path[::-1]


def point_segment_frame(p: Point, a: Point, b: Point) -> tuple[float, float]:
    """(along, across) of ``p`` relative to segment ab: along in metres from a, across signed."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    l = math.hypot(dx, dy)
    if l < EPS:
        return (0.0, dist(p, a))
    ux, uy = dx / l, dy / l
    px, py = p[0] - a[0], p[1] - a[1]
    return (px * ux + py * uy, -px * uy + py * ux)
