"""The vendored connectome is real data and behaves as a quick-reflex substrate."""
from __future__ import annotations

from fly_brain.brain import ConnectomeBrain
from fly_brain.encoders import encode
from fly_brain.perception import Percept
from fly_brain.schemas import DroneState, Goal
from fly_brain.motor import MotorLayer


def test_subset_is_a_real_hemibrain_graph() -> None:
    b = ConnectomeBrain()
    st = b.status()
    assert st.loaded and st.backend == 'connectome'
    assert st.n_neurons > 5_000 and st.n_edges > 500_000
    assert b.subset.ids_for('Giant Fiber') and b.subset.ids_for('LC4')
    assert b.subset.ids_for('LPLC2') and b.subset.ids_for('pC1a')
    assert b.subset.ids_for('DNp13')
    assert b.subset.metadata['source'].startswith('hemibrain')


def test_strong_loom_fires_the_giant_fibre_and_empty_sky_does_not() -> None:
    b = ConnectomeBrain()
    quiet = b.step(encode(Percept(has_target=False)), 50)
    assert quiet.spikes['gf'] == 0
    strong = b.step(
        encode(Percept(has_target=True, size_deg=40, expansion_dps=200, distance_m=5)), 50
    )
    assert strong.spikes['gf'] > 0
    weak = ConnectomeBrain().step(
        encode(Percept(has_target=True, size_deg=12, expansion_dps=3, distance_m=10)), 50
    )
    assert weak.spikes['gf'] == 0


def test_arousal_drives_the_reproductive_readout_and_commitment() -> None:
    b = ConnectomeBrain()
    p = Percept(has_target=False, arousal=1.0)
    out = b.step(encode(p), 50)
    assert out.spikes['courtship'] >= 1
    m = MotorLayer()
    st = DroneState(connected=True, position=(0.0, 0.0, 15.0), yaw_rad=0.0, altitude_m=15.0)
    m.set_goal(Goal(x=20.0, y=0.0, z=15.0, arousal=1.0))
    cmd = m.decide(p, out, st)
    assert cmd.state in ('NAVIGATE', 'HOLD', 'ESCAPE')
    assert m.courtship_drive > 0.0
