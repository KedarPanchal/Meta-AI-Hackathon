"""
The lidar as a compound eye: range clusters become angular threats for the loom tracker.

The X3 carries a 3-D lidar for the brain (``lidar_3d`` in ``models/x3_uav/model.sdf``: 32
planes over +-30 degrees of elevation, bridged as a ``sensor_msgs/PointCloud2`` on
``/drone/lidar3d_pointcloud``). Each scan is placed in the world with the odometry attitude,
cut to a vertical band around the airframe, and split into connected clusters on a polar
grid. Each cluster is then described the way the fly's eye would describe it: bearing,
elevation, angular width, and -- the thing a camera cannot give -- range. The tracker
differentiates angular width per object, so an object closing on the drone grows and looms
exactly as a camera blob does, and the escape log gets a true time-to-contact.

Tracks live in the **world frame**. The drone yaws to face its direction of travel, so two
scans 100 ms apart see the same trunk at different bearings; associating in body coordinates
would hand out a fresh id on every turn and the expansion history with it.

Three things a lidar on a multirotor does that a loom detector must not believe:

* **It sweeps the ground.** The airframe tilts to accelerate, or sits on a slope, and the
  lower planes cut the hillside somewhere in range. As the tilt changes that cut moves, metres
  per second, which is a textbook loom. The odometry attitude places every return in the
  world, and returns more than ``max_below_m`` under the airframe (or ``max_above_m`` over
  it, the canopy) are dropped before anything is clustered.
* **A single plane misses everything at its own level once the nose drops.** A multirotor
  cruises 10 degrees nose-down, so a horizontal scan looks at the ground 80 m ahead and a
  sphere coming at the airframe stays above it until the last metre. That is why the brain
  has its own 3-D lidar rather than the single-plane one ``drone_controller`` uses.
* **Its clusters breathe.** The return count of a cluster changes by a beam or two from scan
  to scan, and branches split a tree into fragments that merge again. Differentiated at 10 Hz
  that is several degrees per second of fake expansion. So a track's physical width is
  measured over its first scans and then **frozen**, and the angular size handed to the
  tracker is computed from that width and the (median of recent) range, which the lidar
  measures to a centimetre. Only a change in range can then make an object grow -- and a
  range step larger than a scan's worth of motion starts a fresh track instead.
* **Scenery steps about anyway.** Two trees' clusters merge and split as the beams fall, and
  the climb out of a canopy brings a different branch into the band every scan. No amount of
  smoothing makes a static scene loom-proof, so the lidar reflex is for **moving** things:
  a track is moving when its world-frame centroid has moved at least ``moving_mps`` on each
  of the last ``motion_window`` scans. A thrown sphere qualifies in three scans; a tree,
  whatever its cluster does, never does. A tree the drone flies at is navigation's problem.

No rclpy here: :func:`pointcloud_xyz` only reads the message's fields, so this file is
testable with a stand-in message.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from fly_brain.perception import Threat

BEAM_RAD = 2.0 * math.pi / 320.0
"""Horizontal angular width of one beam of the 3-D lidar (320 beams over 360 degrees)."""


def pointcloud_xyz(msg):
    """``sensor_msgs/PointCloud2`` -> (N, 3) float array of the finite x, y, z points."""
    import numpy as np

    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in ('x', 'y', 'z')):
        return np.zeros((0, 3))
    endian = '>' if msg.is_bigendian else '<'
    dtype = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [endian + 'f4'] * 3,
        'offsets': [int(fields[k].offset) for k in ('x', 'y', 'z')],
        'itemsize': int(msg.point_step),
    })
    data = bytes(msg.data)
    rows = []
    for r in range(int(msg.height)):
        start = r * int(msg.row_step)
        n = int(msg.width)
        if start + n * int(msg.point_step) > len(data):
            n = max(0, (len(data) - start) // int(msg.point_step))
        if n:
            rows.append(np.frombuffer(data, dtype=dtype, count=n, offset=start))
    if not rows:
        return np.zeros((0, 3))
    buf = np.concatenate(rows)
    pts = np.stack([buf['x'], buf['y'], buf['z']], axis=1).astype(float)
    return pts[np.isfinite(pts).all(axis=1)]


def ground_mask(points_xyz, roll_rad: float, pitch_rad: float, max_below_m: float,
                max_above_m: float = float('inf')):
    """
    True for returns within ``max_below_m`` below and ``max_above_m`` above the airframe.

    Only the vertical component of the body-to-world rotation matters (ZYX Euler, so the
    third row is ``(-sin p, cos p sin r, cos p cos r)``); yaw does not move a point up or down.
    """
    import numpy as np

    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or len(pts) == 0:
        return np.zeros(0, dtype=bool)
    sp, cp = math.sin(pitch_rad), math.cos(pitch_rad)
    sr, cr = math.sin(roll_rad), math.cos(roll_rad)
    z_world = -sp * pts[:, 0] + cp * sr * pts[:, 1] + cp * cr * pts[:, 2]
    return (z_world >= -abs(max_below_m)) & (z_world <= abs(max_above_m))


@dataclass
class Cluster:
    """One connected group of returns, in the sensor (body FLU) frame."""

    bearing_deg: float
    """+ is RIGHT of the nose."""
    elevation_deg: float
    """+ is UP, of the cluster's centroid."""
    span_deg: float
    """Horizontal angular width, including half a beam each side."""
    distance_m: float
    """Nearest return in the cluster, straight-line (3-D) range."""
    width_m: float
    """Physical width implied by span and range."""
    height_m: float
    """Vertical extent of the returns."""
    n_points: int
    body_xy: tuple[float, float]
    """Centroid, x forward, y left."""
    up: float = 0.0
    """Vertical component of the unit vector from the sensor to the centroid, + up."""


def cluster_points(
    points_xyz,
    max_range_m: float = 20.0,
    min_range_m: float = 0.6,
    min_points: int = 2,
    beam_rad: float = BEAM_RAD,
    range_bin_m: float = 0.5,
) -> list[Cluster]:
    """
    Connected components of the returns on a polar grid: one beam by ``range_bin_m``.

    ``points_xyz`` is (N, 3), or (N, 2) for a single plane, in the sensor frame. Returns
    beyond ``max_range_m`` (horizontal) are treated as no return: the sensor's own maximum is
    30 m, and anything that far is not a threat yet. Neighbouring beams within two range bins
    of each other belong together, whatever plane they came from, so several planes hitting
    the same trunk make one cluster, two objects at the same bearing and different ranges make
    two, and a far wall stays whole however sparse its returns. The seam directly behind the
    drone is stitched.
    """
    import numpy as np
    from scipy import ndimage

    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or len(pts) == 0:
        return []
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    rh = np.hypot(pts[:, 0], pts[:, 1])
    keep = (rh > min_range_m) & (rh < max_range_m)
    pts, rh = pts[keep], rh[keep]
    if len(pts) == 0:
        return []
    az = np.arctan2(pts[:, 1], pts[:, 0])
    n_az = max(8, int(round(2.0 * math.pi / beam_rad)))
    n_r = int(math.ceil(max_range_m / range_bin_m)) + 1
    ia = (np.floor((az + math.pi) / (2.0 * math.pi) * n_az).astype(int)) % n_az
    ir = np.clip((rh / range_bin_m).astype(int), 0, n_r - 1)
    grid = np.zeros((n_az, n_r), dtype=bool)
    grid[ia, ir] = True
    # dilate one bin each way, then 8-connect: a single missing beam (a dropout, a sliver of
    # gap between two objects) does not split a cluster, and beams within two range bins join
    grid = ndimage.binary_dilation(grid, structure=np.ones((3, 3), dtype=bool))
    labels, n_labels = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
    if n_labels == 0:
        return []
    # stitch the seam: the last beam is next to the first one
    parent = list(range(n_labels + 1))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    first, last = labels[0], labels[-1]
    for j in np.flatnonzero(first):
        for k in range(max(0, j - 2), min(n_r, j + 3)):
            if last[k]:
                a, b = find(int(first[j])), find(int(last[k]))
                if a != b:
                    parent[b] = a
    lab = np.array([find(int(v)) for v in labels[ia, ir]])
    order = np.argsort(lab, kind='stable')
    lab_sorted = lab[order]
    bounds = np.flatnonzero(np.diff(lab_sorted)) + 1
    out: list[Cluster] = []
    for idx in np.split(order, bounds):
        if len(idx) < min_points:
            continue
        p = pts[idx]
        r = rh[idx]
        a = az[idx]
        mean_az = math.atan2(float(np.mean(np.sin(a))), float(np.mean(np.cos(a))))
        rel = (a - mean_az + math.pi) % (2.0 * math.pi) - math.pi
        span = float(rel.max() - rel.min()) + beam_rad
        r3 = np.sqrt(r * r + p[:, 2] * p[:, 2])
        dist = float(r3.min())
        zc, rc = float(p[:, 2].mean()), float(r.mean())
        out.append(Cluster(
            bearing_deg=-math.degrees(mean_az),          # sensor +y is left; bearing + is right
            elevation_deg=math.degrees(math.atan2(zc, rc)),
            span_deg=math.degrees(span),
            distance_m=dist,
            width_m=2.0 * dist * math.tan(min(span, math.pi * 0.99) / 2.0),
            height_m=float(p[:, 2].max() - p[:, 2].min()),
            n_points=int(len(idx)),
            body_xy=(float(p[:, 0].mean()), float(p[:, 1].mean())),
            up=zc / max(math.hypot(zc, rc), 1e-6),
        ))
    out.sort(key=lambda c: c.distance_m)
    return out


@dataclass
class _LidarTrack:
    world_xyz: tuple[float, float, float]
    t: float
    widths: list[float] = field(default_factory=list)
    """Physical widths seen while settling; the median becomes ``width`` and stays."""
    width: float | None = None
    """Frozen once ``width_window`` scans have been seen."""
    ranges: list[float] = field(default_factory=list)
    """Recent nearest ranges (``range_window`` of them, median taken)."""
    dz: float = 0.0
    """Vertical speed of the centroid over the last scan, world frame."""
    steps: list[tuple[float, float, float]] = field(default_factory=list)
    """World-frame centroid displacements over the last few scans, (dx, dy, dt). An object is
    moving when every one of the last ``motion_window`` is faster than ``moving_mps``, their
    speeds agree within a factor ``speed_ratio_max`` and their directions agree (a thrown
    sphere flies straight at one speed); a static tree whose cluster merges with its
    neighbour, or whose branches enter the band one by one as the drone climbs, jumps about
    at speeds that bear no relation to each other."""


@dataclass
class LidarThreatDetector:
    """
    Clusters with stable ids across scans, as :class:`Threat` objects for the loom tracker.

    ``max_width_m`` is the one filter with a story: a lidar on a multirotor that pitches
    forward to accelerate at low altitude can sweep the ground, which comes back as one
    enormous arc that grows as the aircraft tilts. Nothing that is 8 m wide is a bird.
    """

    max_range_m: float = 20.0
    max_width_m: float = 8.0
    max_below_m: float = 2.0
    """Returns further below the airframe than this are the ground, not objects."""
    max_above_m: float = 4.0
    """Returns further above it are the canopy, or a bird that is not coming down."""
    min_points: int = 2
    range_bin_m: float = 0.5
    assoc_radius_m: float = 1.2
    """Base association radius in the world frame..."""
    assoc_frac: float = 0.05
    """...plus this fraction of the range, because a far cluster's centroid wanders. Was 1.5
    and 0.15, which at 12 m let a track hop to the next tree."""
    track_ttl_s: float = 0.6
    width_window: int = 2
    """Scans over which a track's width is measured before it is frozen. Until then the threat
    is reported unsettled and the tracker does not differentiate it: a width estimate that is
    still moving looked exactly like a loom and fired the giant fibre at startup."""
    range_window: int = 1
    """Was a median of three, a guard against a static cluster's nearest return stepping as
    fragments merged; static things no longer loom at all, and a moving sphere's nearest
    return is clean, so the lag is not worth paying."""
    max_range_step_m: float = 1.5
    """A candidate whose nearest range differs from the track's by more than this a scan later
    is a different thing (a merge, a split), however close its centroid: a fresh track."""
    motion_window: int = 2
    moving_mps: float = 2.5
    """Centroid speed, in the world frame, that every one of the last ``motion_window``
    displacements must show before a track counts as moving. Birds and drones do this;
    nothing in a forest does."""
    speed_ratio_max: float = 2.0
    """...and the fastest of those displacements may be at most this times the slowest..."""
    min_direction_cos: float = 0.5
    """...and consecutive displacements must point within 60 degrees of each other."""
    source: str = 'lidar'
    last_clusters: list[Cluster] = field(default_factory=list)
    _tracks: dict[int, _LidarTrack] = field(default_factory=dict)
    _next_id: int = 0

    def update(
        self, points, drone_xy: tuple[float, float], yaw_rad: float, t: float,
        roll_rad: float = 0.0, pitch_rad: float = 0.0, drone_z: float = 0.0,
    ) -> list[Threat]:
        """
        One scan in, the threats it contains out.

        ``points`` is (N, 2) or (N, 3) in the sensor frame; with three columns the attitude
        places them in the world and returns outside the vertical band are dropped first.
        ``yaw_rad`` and ``drone_xy``/``drone_z`` place the clusters in the world for tracking,
        and the world-frame track decides whether a thing is **moving**: only moving things
        are allowed to loom (:class:`fly_brain.perception.Threat.moving`). A tree the drone
        climbs past, or two trees whose clusters merge and split as the beams fall, are
        scenery however their nearest return steps about.
        """
        import numpy as np

        pts = np.asarray(points, dtype=float)
        if pts.ndim == 2 and pts.shape[1] >= 3:
            pts = pts[ground_mask(pts[:, :3], roll_rad, pitch_rad, self.max_below_m,
                                  self.max_above_m)][:, :3]
        clusters = [
            c for c in cluster_points(pts, max_range_m=self.max_range_m,
                                      range_bin_m=self.range_bin_m, min_points=self.min_points)
            if c.width_m <= self.max_width_m
        ]
        self.last_clusters = clusters
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        threats: list[Threat] = []
        claimed: set[int] = set()
        for cl in clusters:
            bx, by = cl.body_xy
            wx, wy = drone_xy[0] + c * bx - s * by, drone_xy[1] + s * bx + c * by
            wz = drone_z + cl.distance_m * cl.up
            radius = self.assoc_radius_m + self.assoc_frac * cl.distance_m
            best, best_d = -1, radius * radius
            for tid, tr in self._tracks.items():
                if tid in claimed:
                    continue
                if abs(cl.distance_m - tr.ranges[-1]) > self.max_range_step_m:
                    continue        # a merge or a split, not the same thing a scan later
                d = (wx - tr.world_xyz[0]) ** 2 + (wy - tr.world_xyz[1]) ** 2
                if d < best_d:
                    best, best_d = tid, d
            prev = self._tracks.get(best) if best >= 0 else None
            if prev is None:
                self._next_id += 1
                best = self._next_id
            claimed.add(best)
            tr = _LidarTrack(world_xyz=(wx, wy, wz), t=t)
            if prev is not None:
                tr.widths, tr.width, tr.ranges, tr.steps = (
                    prev.widths, prev.width, prev.ranges, prev.steps
                )
                dt = t - prev.t
                if dt > 1e-3:
                    tr.steps = (tr.steps + [(wx - prev.world_xyz[0], wy - prev.world_xyz[1], dt)])[-self.motion_window:]
                    tr.dz = (wz - prev.world_xyz[2]) / dt
            if tr.width is None:
                tr.widths = (tr.widths + [cl.width_m])[-self.width_window:]
                if len(tr.widths) >= self.width_window:
                    tr.width = sorted(tr.widths)[len(tr.widths) // 2]
            tr.ranges = (tr.ranges + [cl.distance_m])[-self.range_window:]
            self._tracks[best] = tr
            settled = tr.width is not None
            width = tr.width if settled else cl.width_m
            rng = sorted(tr.ranges)[len(tr.ranges) // 2]
            size = 2.0 * math.degrees(math.atan(width / (2.0 * max(rng, 0.05))))
            moving, speed = self._is_moving(tr.steps)
            velocity = None
            if moving:
                # mean world velocity over the window, turned into the body frame
                vwx = sum(dx / dt for dx, _, dt in tr.steps) / len(tr.steps)
                vwy = sum(dy / dt for _, dy, dt in tr.steps) / len(tr.steps)
                velocity = (c * vwx + s * vwy, -s * vwx + c * vwy, tr.dz)
            threats.append(Threat(
                bearing_deg=cl.bearing_deg, elevation_deg=cl.elevation_deg, size_deg=size,
                track_id=best, source=self.source, distance_m=rng, t=t, settled=settled,
                moving=moving, velocity_body=velocity, height_m=cl.height_m,
                label=f'{width:.1f} m wide, {cl.n_points} pts'
                      + (f', {speed:.1f} m/s' if moving else ', static'),
            ))
        for tid in [k for k, tr in self._tracks.items() if t - tr.t > self.track_ttl_s]:
            self._tracks.pop(tid, None)
        return threats

    def _is_moving(self, steps: list[tuple[float, float, float]]) -> tuple[bool, float]:
        """Straight, steady and fast enough over the whole window; else scenery."""
        if len(steps) < self.motion_window:
            return False, 0.0
        speeds = [math.hypot(dx, dy) / dt for dx, dy, dt in steps]
        if min(speeds) < self.moving_mps or max(speeds) > self.speed_ratio_max * min(speeds):
            return False, 0.0
        for (ax, ay, _), (bx, by, _) in zip(steps, steps[1:]):
            na, nb = math.hypot(ax, ay), math.hypot(bx, by)
            if (ax * bx + ay * by) / max(na * nb, 1e-9) < self.min_direction_cos:
                return False, 0.0
        return True, sum(speeds) / len(speeds)

    def reset(self) -> None:
        self._tracks.clear()
        self.last_clusters = []
