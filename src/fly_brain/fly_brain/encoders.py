"""
Percept -> Poisson drive, one ``(left_hz, right_hz)`` pair per input channel.

**Rates are small on purpose.** FLYCNS anchors its own loom protocol at
``PEAK_HZ_DEFAULT = 5.0`` Hz per stimulated cell, chosen so the giant fibre fires the one or
two spikes von Reyn et al. 2014 measured. Driving these populations at tens or hundreds of Hz
makes DNp01 fire continuously, which destroys the "a GF spike is a trigger event" property the
whole escape reflex rests on. Keep ``peak_hz`` in single digits unless a calibration says
otherwise.

The left/right split is the fly's two eyes: a target at bearing ``b`` drives the right
population by ``(1 + sin b) / 2`` and the left by ``(1 - sin b) / 2``, so straight ahead is
balanced and 90 degrees to the right is fully lateralised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os

from .schemas import Percept

# FLYCNS's own anchor (flycns/protocols.py PEAK_HZ_DEFAULT), overridable for calibration.
PEAK_HZ_DEFAULT = float(os.environ.get('FLYDRONE_PEAK_HZ', '5.0'))

# Relative loom sensitivity of the two giant-fibre inputs, from flycns/protocols.py
# LOOM_TUNING (Turner et al. 2022): LC4 and LPLC2 are both at the top of the ordinal scale.
LOOM_TUNING = {'loom_speed': 1.0, 'loom_size': 1.0}

# Calibrated on the toy graph: a 0.5 m object closing at 5 m/s from 6 m, 50 ms tick, sweeping
# this gain and recording the time-to-contact when DNp01 first spiked.
#
#     gain | b=0    b=-30  b=+30  b=-60  b=+60
#     x12  | 0.90s  1.10s  0.75s  1.10s  0.90s   <- marginal at +30
#     x16  | 1.10s  1.10s  0.90s  1.10s  0.90s   <- chosen
#     x20  | 1.10s  1.10s  1.15s  1.10s  1.10s
#
# The escape needs tau_trigger >= latency + sqrt(2 d / a), about 0.75 s for 0.9 m of clearance
# at 5 m/s^2 with 0.15 s of latency, so x16 is the smallest gain that clears it at every
# bearing with margin. Going higher buys little and fires at harmless things.
#
# Two caveats worth keeping in mind:
#   * The trigger distance depends on how long the object has been in view, because the
#     membrane integrates: the same object starting at 6 m triggers earlier than at 5 m.
#   * The bearings are not symmetric. The eye split drives one half of the population harder,
#     and the toy graph's two sides are not identical populations.
# Re-measure on the real connectome; the toy graph's giant-fibre input is not the real one.
LOOM_GAIN_DEFAULT = float(os.environ.get('FLYDRONE_LOOM_GAIN', '16.0'))


@dataclass
class EncoderParams:
    """Saturation points: the stimulus value that maps to the full rate."""

    peak_hz: float = PEAK_HZ_DEFAULT
    expansion_sat_dps: float = 400.0
    """Angular expansion rate (deg/s) giving the full LC4 rate."""
    size_sat_deg: float = 60.0
    """Angular size (deg) giving the full LPLC2 rate."""
    target_sat_deg: float = 10.0
    """Angular size (deg) of a tracked target giving the full LC10a rate."""
    sound_floor_db: float = -60.0
    """dBFS at which the auditory channel starts responding."""
    sound_range_db: float = 40.0
    """dB above the floor giving the full Johnston's-organ rate."""
    gains: dict[str, float] = field(
        default_factory=lambda: {
            'loom_speed': LOOM_GAIN_DEFAULT,
            'loom_size': LOOM_GAIN_DEFAULT,
            'target': 1.0,
            'sound': 1.0,
            'arousal': 1.0,
        }
    )
    """Per-channel multiplier, the knob that ``test_reflex`` calibrates."""


def _sat(value: float, ceiling: float) -> float:
    """Clamp ``value / ceiling`` to 0..1."""
    if ceiling <= 0:
        return 0.0
    return max(0.0, min(1.0, value / ceiling))


def eye_split(bearing_deg: float) -> tuple[float, float]:
    """Fraction of a visual drive reaching the (left, right) population."""
    s = math.sin(math.radians(max(-90.0, min(90.0, bearing_deg))))
    return (1.0 - s) / 2.0, (1.0 + s) / 2.0


def encode(
    percept: Percept, params: EncoderParams | None = None
) -> dict[str, tuple[float, float]]:
    """Turn one percept into per-channel ``(left_hz, right_hz)`` Poisson rates."""
    p = params or EncoderParams()
    peak = p.peak_hz
    left, right = eye_split(percept.bearing_deg)
    drive: dict[str, tuple[float, float]] = {}

    def put(channel: str, unit: float, lateral: bool = True) -> None:
        gain = p.gains.get(channel, 1.0) * LOOM_TUNING.get(channel, 1.0)
        hz = peak * gain * max(0.0, min(1.0, unit))
        drive[channel] = (hz * left, hz * right) if lateral else (hz, hz)

    seen = percept.has_target
    # Looming: expansion rate drives LC4, absolute size drives LPLC2.
    put('loom_speed', _sat(percept.expansion_dps, p.expansion_sat_dps) if seen else 0.0)
    put('loom_size', _sat(percept.size_deg, p.size_sat_deg) if seen else 0.0)
    # Small-object tracking for pursuit.
    put('target', _sat(percept.size_deg, p.target_sat_deg) if seen else 0.0)
    # Hearing is not lateralised here: the rotors swamp any interaural cue.
    put('sound', _sat(percept.sound_db - p.sound_floor_db, p.sound_range_db), lateral=False)
    # The mission gate: tonic, bilateral, no direction in it.
    put('arousal', max(0.0, min(1.0, percept.arousal)), lateral=False)
    return drive


def total_hz(drive: dict[str, tuple[float, float]]) -> float:
    """Sum of every channel's rate, for logging and sanity checks."""
    return float(sum(left + right for left, right in drive.values()))
