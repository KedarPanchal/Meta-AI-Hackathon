"""
The brain backends: a real, self-contained connectome subset and a cheap stand-in.

Both expose the same two calls, so everything above them (encoders, motor, ROS
node, MCP tools) is identical whichever is loaded::

    out = brain.step({"loom_speed": (0.0, 400.0), ...}, tick_ms=20)
    brain.status()

The real backend is :class:`fly_brain.connectome.ConnectomeBrain`.  It loads a
vendored gzipped subgraph of the public hemibrain v1.2.1 connectome and runs a
small NumPy LIF network.  There is no Brian2 import, no worker-thread SIGINT
trap, and no sibling ``FLYCNS`` checkout.  ``MockBrain`` remains a linear
stand-in for tests that do not need spike timing.
"""
from __future__ import annotations

from collections import deque
import time
from typing import Any

from .circuit import Circuit, INPUT_CHANNELS
from .config import Settings, settings
from .connectome import ConnectomeBrain
from .schemas import BrainOut, BrainStatus

# MockBrain trigger points, in the rates fly_brain.encoders emits.  These are
# deliberately tied to the encoder's scale; if LOOM_MIN_DPS, LOOM_GAIN or
# PEAK_HZ changes, update the mock and its comment or the mock and the
# connectome stop predicting each other.
MOCK_GF_THRESHOLD_HZ = 7.0
MOCK_GF_DOUBLE_HZ = 16.0
MOCK_SOUND_THRESHOLD_HZ = 2.0


class MockBrain:
    """A rate stand-in: spikes come straight from the drive, no network."""

    backend = 'mock'

    def __init__(self, circuit: Circuit | None = None, cfg: Settings | None = None):
        self.cfg = cfg or settings()
        self.circuit = circuit or Circuit()
        self.circuit.resolve(
            ['LC4', 'LPLC2', 'LC10a', 'JO-B1_a', 'JO-B1_c', 'pC1_1b', 'DNp01', 'TTMn', 'PSI']
        )
        self.ticks = 0
        self.ablated: list[str] = []
        self._rt: deque[float] = deque(maxlen=50)

    def step(self, drive: dict[str, tuple[float, float]],
             tick_ms: float | None = None) -> BrainOut:
        tick_ms = float(tick_ms or self.cfg.brain_tick_ms)
        t0 = time.perf_counter()
        secs = tick_ms / 1000.0

        def lr(ch: str) -> tuple[float, float]:
            return drive.get(ch, (0.0, 0.0))

        loom = sum(lr('loom_speed')) + sum(lr('loom_size'))
        sound = sum(lr('sound'))
        arousal = sum(lr('arousal'))
        gf = 0
        if loom > MOCK_GF_THRESHOLD_HZ:
            gf = 1 + int(loom > MOCK_GF_DOUBLE_HZ)
        if sound > MOCK_SOUND_THRESHOLD_HZ and loom > MOCK_GF_THRESHOLD_HZ * 0.75:
            gf = max(gf, 1)
        jump = gf
        power = int(gf > 0)
        tl, tr = lr('target')
        steer_l = int(tl * secs * 0.5)
        steer_r = int(tr * secs * 0.5)
        fwd = int(arousal * secs * 0.5)
        courtship = int(arousal * secs * 2.0)
        self.ticks += 1
        wall = (time.perf_counter() - t0) * 1000.0
        self._rt.append(tick_ms / max(wall, 1e-6))
        return BrainOut(
            tick_ms=tick_ms,
            spikes={'gf': gf, 'jump': jump, 'flight_power': power,
                    'steer': steer_l + steer_r, 'forward': fwd, 'courtship': courtship},
            spikes_lr={
                'gf': (gf, gf), 'jump': (jump, jump), 'flight_power': (power, power),
                'steer': (steer_l, steer_r), 'forward': (fwd, fwd),
                'courtship': (courtship, courtship),
            },
            wall_ms=wall,
        )

    def status(self) -> BrainStatus:
        return BrainStatus(
            backend='mock', loaded=True, data_dir='(mock)',
            tick_ms=self.cfg.brain_tick_ms,
            realtime_factor=(sum(self._rt) / len(self._rt)) if self._rt else 0.0,
            ticks=self.ticks, ablated=list(self.ablated),
        )

    def stimulate(self, types: list[str], hz: float = 20.0, ms: float = 200.0) -> dict:
        return {'backend': 'mock', 'note': 'the mock has no network to stimulate', 'spikes': {}}


def make_brain(backend: str | None = None, circuit: Circuit | None = None,
               cfg: Settings | None = None, ablate: list[str] | None = None,
               prune_hops: int | None = None):
    """Build whichever backend the environment asks for."""
    cfg = cfg or settings()
    backend = (backend or cfg.backend or 'connectome').lower()
    if backend in ('connectome', 'hemibrain'):
        return ConnectomeBrain(circuit=circuit, cfg=cfg, ablate=ablate, prune_hops=prune_hops)
    if backend == 'mock':
        return MockBrain(circuit=circuit, cfg=cfg)
    raise ValueError(f'unknown BRAIN_BACKEND {backend!r}; choose connectome or mock')
