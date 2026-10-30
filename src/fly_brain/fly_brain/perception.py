"""
Camera and microphone front ends, plus the synthetic loom used to calibrate the reflex.

A monocular camera gives no range. Everything here is therefore angular: bearing, angular
size, and the rate of change of angular size. That is also what the fly's lobula columnar
neurons see, so the loss is not as bad as it sounds -- but it does mean "distance" is only
available when something else supplies it (Gazebo ground truth, a depth camera, or a known
object size).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

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
class _Track:
    size_deg: float
    t: float


@dataclass
class LoomTracker:
    """
    Turns detections into a percept, differentiating angular size to get expansion rate.

    Keeps one previous sample per track id so expansion is per object, not per frame: without
    that, two objects at different ranges alternate and produce enormous spurious expansion.
    """

    hfov_deg: float = 90.0
    smooth: float = 0.5
    """Low-pass on the expansion estimate, 0..1. Differentiating pixels is noisy."""
    history: dict[int, _Track] = field(default_factory=dict)
    expansion: dict[int, float] = field(default_factory=dict)

    def update(self, boxes: list[Box], t: float) -> Percept:
        """Pick the most threatening box and describe it. Empty input gives an empty percept."""
        if not boxes:
            self.history.clear()
            self.expansion.clear()
            return Percept(has_target=False)
        best: Box | None = None
        best_size = -1.0
        for b in boxes:
            s = size_of(b, self.hfov_deg)
            prev = self.history.get(b.track_id)
            if prev is not None and t > prev.t:
                raw = (s - prev.size_deg) / (t - prev.t)
                old = self.expansion.get(b.track_id, 0.0)
                self.expansion[b.track_id] = old + self.smooth * (raw - old)
            self.history[b.track_id] = _Track(size_deg=s, t=t)
            # "most threatening" = biggest, which for a constant physical size means nearest
            if s > best_size:
                best, best_size = b, s
        assert best is not None
        bearing, elevation = bearing_of(best, self.hfov_deg)
        live = {b.track_id for b in boxes}
        for gone in [k for k in self.history if k not in live]:
            self.history.pop(gone, None)
            self.expansion.pop(gone, None)
        return Percept(
            has_target=True,
            bearing_deg=bearing,
            elevation_deg=elevation,
            size_deg=best_size,
            expansion_dps=max(0.0, self.expansion.get(best.track_id, 0.0)),
        )


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
