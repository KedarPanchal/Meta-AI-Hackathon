"""The encoders: LPLC2 only drives while the object grows; the approach stimulus is unchanged."""
from __future__ import annotations

from fly_brain.encoders import EncoderParams, encode
from fly_brain.perception import approach_percepts
from fly_brain.schemas import Percept


def test_static_object_drives_no_loom_channel() -> None:
    d = encode(Percept(has_target=True, size_deg=30.0, expansion_dps=0.0), EncoderParams())
    assert d['loom_speed'] == (0.0, 0.0)
    assert d['loom_size'] == (0.0, 0.0)
    assert sum(d['target']) > 0                      # it is still a target to track


def test_growing_object_drives_both() -> None:
    d = encode(Percept(has_target=True, size_deg=30.0, expansion_dps=50.0), EncoderParams())
    assert sum(d['loom_speed']) > 0 and sum(d['loom_size']) > 0


def test_jitter_is_not_a_loom() -> None:
    d = encode(Percept(has_target=True, size_deg=30.0, expansion_dps=1.5), EncoderParams())
    assert d['loom_speed'] == (0.0, 0.0) and d['loom_size'] == (0.0, 0.0)


def test_approach_drive_is_lateralised_and_positive() -> None:
    p = EncoderParams()
    for percept in approach_percepts(distance_m=6.0, speed_mps=5.0, bearing_deg=30.0)[1:]:
        d = encode(percept, p)
        left, right = d['loom_size']
        assert right > left > 0.0                    # target to the right drives the right eye
