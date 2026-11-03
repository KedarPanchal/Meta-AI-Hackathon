"""
Camera, lidar and microphone front ends, plus the synthetic loom used to calibrate the reflex.

A monocular camera gives no range. Everything here is therefore angular: bearing, angular
size, and the rate of change of angular size. That is also what the fly's lobula columnar
neurons see, so the loss is not as bad as it sounds -- but it does mean "distance" is only
available when something else supplies it (the lidar, Gazebo ground truth, or a known object
size).

Every sensor ends in a :class:`Threat`: one object in angular terms, whatever sensed it. The
camera's pixel boxes and the lidar's range clusters both become threats, so the tracker and
the brain behind it never know which sensor saw the thing. The fly only has eyes; the lidar
stands in for the near-360 degree compound eye that a multirotor's forward camera is not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .schemas import Box, Percept


def bearing_of(box: Box, hfov_deg: float) -> tuple[float, float]:
    """(bearing, elevation) of a box centre in degrees; bearing + is RIGHT, elevation + is UP."""
    cx = box.x + box.w / 2.0
    cy = box.y + box.h / 2.0
    # pinhole: tan(angle) is linear in pixel offset from the principal point
    half = math.tan(math.radians(hfov_deg / 2.0))
    nx = (cx / max(box.image_w, 1.0)) * 2.0 - 1.0
    ny = 1.0 - (cy / max(box.image_h, 1.0)) * 2.0
    aspect = max(box.image_h, 1.0) / max(box.image_w, 1.0)
    bearing = math.degrees(math.atan(nx * half))
    elevation = math.degrees(math.atan(ny * half * aspect))
    return bearing, elevation


def size_of(box: Box, hfov_deg: float) -> float:
    """Angular size of a box, degrees, taking the larger of its two extents."""
    half = math.tan(math.radians(hfov_deg / 2.0))
    frac = max(box.w / max(box.image_w, 1.0), box.h / max(box.image_h, 1.0))
    return 2.0 * math.degrees(math.atan(frac * half))


@dataclass
class Threat:
    """
    One object in angular terms, whatever sensed it.

    ``(source, track_id)`` identifies the object across frames, so the tracker can
    differentiate its size. Two sensors never share a key, so a lidar cluster and a camera
    blob that are the same physical object are tracked twice; the brain only ever hears
    about the more urgent of the two, so that costs nothing.
    """

    bearing_deg: float
    """+ is RIGHT of the nose."""
    elevation_deg: float
    """+ is UP."""
    size_deg: float
    """Angular size, the larger extent."""
    track_id: int
    source: str = 'camera'
    distance_m: float | None = None
    """Range, if the sensor measures it. The camera does not; the lidar does."""
    t: float | None = None
    """When it was sensed. None: the tracker's own clock."""
    settled: bool = True
    """False while the sensor's own size estimate is still converging (a new lidar track):
    the tracker remembers the sample but does not differentiate it, so the convergence is
    not mistaken for a loom."""
    velocity_body: tuple[float, float, float] | None = None
    """The object's own velocity in body FLU, m/s, if the sensor tracks it."""
    moving: bool | None = None
    """Whether the object moves in the world, if the sensor can tell (the lidar tracks in the
    world frame; a camera cannot). ``False`` is scenery: it may be tracked and may be the
    biggest thing in view, but it never looms. A tree does not come at a fly; the fly flies
    at the tree, and that is navigation's business, not the escape reflex's."""
    label: str = ''

    @property
    def key(self) -> tuple[str, int]:
        return (self.source, self.track_id)


def threat_from_box(
    box: Box, hfov_deg: float, source: str = 'camera', t: float | None = None
) -> Threat:
    """A camera detection as a threat: angles from the pinhole, no range."""
    bearing, elevation = bearing_of(box, hfov_deg)
    return Threat(
        bearing_deg=bearing, elevation_deg=elevation, size_deg=size_of(box, hfov_deg),
        track_id=box.track_id, source=source, t=t, label=box.label,
    )


@dataclass
class _Track:
    size_deg: float
    t: float
    settled: bool = True
    n: int = 1
    """Samples seen of this key."""


@dataclass
class LoomTracker:
    """
    Turns detections into a percept, differentiating angular size to get expansion rate.

    Keeps one previous sample per track so expansion is per object, not per frame: without
    that, two objects at different ranges alternate and produce enormous spurious expansion.

    The percept describes the **most urgent** object: the smallest optical time-to-contact
    ``tau = size / expansion`` among the objects that are growing, which is the quantity the
    fly's own loom detectors are tuned to. If nothing is growing, the biggest object, which for
    a constant physical size means the nearest.
    """

    hfov_deg: float = 90.0
    smooth: float = 0.7
    """Low-pass on the expansion estimate, 0..1. Differentiating pixels is noisy, but every
    scan of lag is a tenth of a second of the reflex's budget: 0.7 reaches 97% of a step in
    three samples."""
    min_expansion_dps: float = 2.0
    """Below this an expansion estimate is noise: the object is ranked by size alone."""
    min_samples: int = 3
    """A track is differentiated only from its third sample on. A camera blob's first frames
    are where it merges with a neighbour or gets a fresh id, and both look like a leap in size;
    the lookout tower's roof fired the reflex that way from 40 m."""
    max_tau_s: float = 3.0
    """An object whose optical time-to-contact ``size / expansion`` is longer than this is
    not looming yet: it is reported with zero expansion, so neither giant-fibre input is
    driven and it is tracked as a target only. The climb out of a pine canopy is the case:
    the branches at the drone's level come a metre closer per second of climb, a big thing
    growing slowly, which is scenery to a fly and was an escape to this brain."""
    history: dict[tuple[str, int], _Track] = field(default_factory=dict)
    expansion: dict[tuple[str, int], float] = field(default_factory=dict)
    last: Threat | None = None
    """The threat the last percept described, for logging."""

    def update(self, boxes: list[Box], t: float) -> Percept:
        """Camera boxes only: the original entry point, kept for the tests and the sim link."""
        return self.update_threats([threat_from_box(b, self.hfov_deg) for b in boxes], t)

    def update_threats(self, threats: list[Threat], t: float) -> Percept:
        """Pick the most urgent threat and describe it. Empty input gives an empty percept."""
        if not threats:
            self.history.clear()
            self.expansion.clear()
            self.last = None
            return Percept(has_target=False)
        for th in threats:
            k = th.key
            tt = th.t if th.t is not None else t
            prev = self.history.get(k)
            n = 1 if prev is None else prev.n + 1
            if prev is None or not th.settled or not prev.settled or n < self.min_samples:
                # first sight, too few samples, or the sensor's size estimate is (or just was)
                # still converging: remember the sample, differentiate nothing
                self.history[k] = _Track(size_deg=th.size_deg, t=tt, settled=th.settled, n=n)
                self.expansion.pop(k, None)
            elif tt > prev.t:
                raw = (th.size_deg - prev.size_deg) / (tt - prev.t)
                old = self.expansion.get(k, 0.0)
                self.expansion[k] = old + self.smooth * (raw - old)
                self.history[k] = _Track(size_deg=th.size_deg, t=tt, n=n)
            else:
                self.history[k] = _Track(size_deg=prev.size_deg, t=prev.t, n=n)
            # tt <= prev.t: the same sample again; keep the estimate we have
        best = min(threats, key=self._urgency)
        live = {th.key for th in threats}
        for gone in [k for k in self.history if k not in live]:
            self.history.pop(gone, None)
            self.expansion.pop(gone, None)
        self.last = best
        exp = self.expansion_of(best)
        return Percept(
            has_target=True,
            bearing_deg=best.bearing_deg,
            elevation_deg=best.elevation_deg,
            size_deg=best.size_deg,
            expansion_dps=exp,
            distance_m=best.distance_m,
            velocity_body=best.velocity_body,
        )

    def _urgency(self, th: Threat) -> tuple[int, float]:
        """Sort key: objects coming within ``max_tau_s`` by time-to-contact, then size."""
        exp = self.expansion_of(th)
        if exp >= self.min_expansion_dps:
            return (0, th.size_deg / exp)
        return (1, -th.size_deg)

    def expansion_of(self, th: Threat) -> float:
        """The expansion the brain is told about: zero for scenery and for slow growth."""
        if th.moving is False:
            return 0.0
        exp = max(0.0, self.expansion.get(th.key, 0.0))
        if exp > 0.0 and th.size_deg > 0.0 and th.size_deg / exp > self.max_tau_s:
            return 0.0                              # growing, but too slowly to be coming
        return exp


@dataclass
class AudioFrontEnd:
    """
    Band level in dBFS with the rotors notched out and an adaptive noise floor.

    A multirotor's own blade-passing frequency and its harmonics dominate the spectrum, so a
    raw level meter reads the aircraft, not the world. Those bins are removed before the level
    is taken, and the floor tracks the quietest recent frames so the channel measures
    *changes* in the acoustic scene.
    """

    blade_hz: float = 120.0
    """Blade-passing frequency: rotor RPM/60 times the number of blades."""
    n_harmonics: int = 6
    notch_width_hz: float = 12.0
    floor_db: float = -60.0
    floor_rate: float = 0.02
    """How fast the adaptive floor follows a quieter scene, per frame."""

    def level_db(self, samples, sample_rate: float) -> float:
        """Measure the non-rotor energy in one audio frame, in dBFS."""
        import numpy as np

        x = np.asarray(samples, dtype=float).ravel()
        if x.size < 16 or sample_rate <= 0:
            return self.floor_db
        if np.issubdtype(np.asarray(samples).dtype, np.integer):
            x = x / 32768.0
        x = x - x.mean()
        spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
        freqs = np.fft.rfftfreq(x.size, 1.0 / sample_rate)
        keep = np.ones_like(spec, dtype=bool)
        for k in range(1, self.n_harmonics + 1):
            keep &= np.abs(freqs - k * self.blade_hz) > self.notch_width_hz
        keep &= freqs > 50.0
        power = float(np.sum(spec[keep] ** 2)) / max(1, keep.sum())
        db = 10.0 * math.log10(power + 1e-20) - 20.0
        db = max(-120.0, min(0.0, db))
        if db < self.floor_db:
            self.floor_db += (db - self.floor_db) * self.floor_rate * 10.0
        else:
            self.floor_db += (db - self.floor_db) * self.floor_rate
        return db


def approach_percepts(
    distance_m: float = 5.0,
    speed_mps: float = 5.0,
    object_size_m: float = 0.5,
    bearing_deg: float = 0.0,
    dt: float = 0.05,
    altitude_m: float = 3.0,
    sound_db: float = -80.0,
) -> list[Percept]:
    """
    Generate a head-on approach, as angles: the stimulus ``test_reflex`` replays.

    Angular size of a sphere of diameter ``d`` at range ``r`` is ``2 atan(d / 2r)``, and the
    expansion rate is its time derivative. Generating the whole trajectory up front keeps the
    calibration reproducible.
    """
    out: list[Percept] = []
    r = distance_m
    t = 0.0
    prev = 2.0 * math.degrees(math.atan(object_size_m / (2.0 * max(r, 1e-3))))
    while r > object_size_m / 2.0:
        size = 2.0 * math.degrees(math.atan(object_size_m / (2.0 * max(r, 1e-3))))
        out.append(
            Percept(
                has_target=True,
                bearing_deg=bearing_deg,
                size_deg=size,
                expansion_dps=max(0.0, (size - prev) / dt),
                distance_m=r,
                altitude_m=altitude_m,
                sound_db=sound_db,
            )
        )
        prev = size
        r -= speed_mps * dt
        t += dt
    return out
