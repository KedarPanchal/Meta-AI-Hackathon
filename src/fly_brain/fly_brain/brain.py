"""
The brain backends: a real FLYCNS spiking network, and a cheap stand-in for tests.

Both expose the same two calls, so everything above them (encoders, motor, ROS node, MCP
tools) is identical whether or not the connectome is loaded:

    out = brain.step({"loom_speed": (0.0, 400.0), ...}, tick_ms=20)
    brain.status()

Things the FLYCNS backend has to respect, each of which cost a debugging session:

* **Import Brian2 on the main thread.** Brian2 installs a SIGINT handler when imported, and
  Python only allows that from the main thread. FastAPI, MCP and ROS callbacks all run on
  worker threads, so :func:`preimport_flycns` must be called from the main thread at startup.
* **``model.stim`` is in bodyId order, not the order you asked for.** Rates are a vector with
  one entry per stimulated *neuron*, so they must be mapped per row, never zipped.
* **Stimulated types leave the network.** CNSModel replaces them with Poisson sources, so a
  type cannot be both driven and read back.
* **Exact-name matching.** See :mod:`fly_brain.circuit`.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
import time
from typing import Any

from .circuit import Circuit, INPUT_CHANNELS, OUTPUT_CHANNELS
from .config import Settings, settings
from .schemas import BrainOut, BrainStatus

_FLYCNS_READY = False
_FLYCNS_ERROR: str | None = None

# MockBrain trigger points, in the rates fly_brain.encoders emits (see MockBrain.step).
MOCK_GF_THRESHOLD_HZ = 8.0
MOCK_GF_DOUBLE_HZ = 24.0
MOCK_SOUND_THRESHOLD_HZ = 2.0


def preimport_flycns(repo: Path | str | None = None) -> bool:
    """
    Import flycns and Brian2 **on the main thread**. Returns True if they are importable.

    Safe to call more than once and safe to call when the clone is missing; the brain only
    needs this before the first worker-thread tick.
    """
    global _FLYCNS_READY, _FLYCNS_ERROR
    if _FLYCNS_READY:
        return True
    repo = Path(repo) if repo else settings().repo
    if repo and str(repo) not in sys.path and repo.exists():
        sys.path.insert(0, str(repo))
    try:
        import brian2  # noqa: F401
        import flycns  # noqa: F401

        from flycns import CNSModel, LIFParams, load_graph  # noqa: F401
    except Exception as exc:
        _FLYCNS_ERROR = f'{type(exc).__name__}: {exc}'
        if isinstance(exc, ValueError) and 'main thread' in str(exc):
            # The documented trap: Brian2 installs a SIGINT handler when imported and Python
            # only allows that on the main thread. Say so, because the bare ValueError gives
            # no hint that the fix is to import it earlier rather than to install anything.
            _FLYCNS_ERROR += (
                ' -- Brian2 can only be imported from the MAIN thread. Call '
                'preimport_flycns() at process start, before any worker thread asks for a tick.'
            )
        return False
    _FLYCNS_ERROR = None
    _FLYCNS_READY = True
    return True


def flycns_import_error() -> str | None:
    """Why the last :func:`preimport_flycns` failed, or None if it has not."""
    return _FLYCNS_ERROR


class MockBrain:
    """
    A rate stand-in: spikes come straight from the drive, no network.

    Exists so the motor layer, the ROS node and every MCP tool can be tested without the
    connectome. It is deliberately linear, so nothing fitted against it means anything.
    """

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

    def step(
        self, drive: dict[str, tuple[float, float]], tick_ms: float | None = None
    ) -> BrainOut:
        tick_ms = float(tick_ms or self.cfg.brain_tick_ms)
        t0 = time.perf_counter()
        secs = tick_ms / 1000.0

        def lr(ch: str) -> tuple[float, float]:
            return drive.get(ch, (0.0, 0.0))

        loom = max(sum(lr('loom_speed')), sum(lr('loom_size')))
        sound = sum(lr('sound'))
        arousal = sum(lr('arousal'))
        # A loom past threshold produces one or two spikes, as the real giant fibre does.
        # The thresholds are in the units the encoder actually emits -- single-digit to tens
        # of Hz, not hundreds -- and are set so this fires at roughly the same range the toy
        # connectome does (about 4.5 m for a 0.5 m object at 5 m/s). If the encoder's peak
        # rate or loom gain changes, these move with it or the mock stops predicting anything.
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
        self.ticks += 1
        wall = (time.perf_counter() - t0) * 1000.0
        self._rt.append(tick_ms / max(wall, 1e-6))
        return BrainOut(
            tick_ms=tick_ms,
            spikes={'gf': gf, 'jump': jump, 'flight_power': power,
                    'steer': steer_l + steer_r, 'forward': fwd},
            spikes_lr={
                'gf': (gf, gf),
                'jump': (jump, jump),
                'flight_power': (power, power),
                'steer': (steer_l, steer_r),
                'forward': (fwd, fwd),
            },
            wall_ms=wall,
        )

    def status(self) -> BrainStatus:
        return BrainStatus(
            backend='mock',
            loaded=True,
            data_dir='(mock)',
            tick_ms=self.cfg.brain_tick_ms,
            realtime_factor=(sum(self._rt) / len(self._rt)) if self._rt else 0.0,
            ticks=self.ticks,
            ablated=list(self.ablated),
        )


class FlycnsBrain:
    """A real connectome-constrained LIF network driven as a closed-loop controller."""

    backend = 'flycns'

    def __init__(
        self,
        circuit: Circuit | None = None,
        cfg: Settings | None = None,
        ablate: list[str] | None = None,
        prune_hops: int | None = None,
    ):
        self.cfg = cfg or settings()
        if not preimport_flycns(self.cfg.repo):
            raise RuntimeError(
                f'flycns is not importable from FLYDRONE_REPO={self.cfg.repo!r} '
                f'({flycns_import_error()}). Set FLYDRONE_REPO to the FLYCNS clone and make '
                'sure brian2 is installed.'
            )
        import numpy as np
        from flycns import CNSModel, LIFParams, load_graph
        from flycns.graph import infer_side

        self._np = np
        data_dir = self.cfg.flycns_data_dir()
        if not Path(data_dir).exists():
            raise RuntimeError(
                f'graph directory {data_dir} does not exist. Run scripts/make_testdata.py in the '
                'FLYCNS clone for the toy graph, or scripts/setup.py for the real connectome.'
            )
        self.data_dir = str(data_dir)
        neurons, edges = load_graph(self.data_dir)

        self.circuit = circuit or Circuit()
        self.circuit.resolve(neurons['type'].dropna().unique())

        self.ablated = list(ablate or [])
        if self.ablated:
            edges = ablate_types(edges, neurons, self.ablated)

        if prune_hops:
            neurons, edges = prune_to_circuit(neurons, edges, self.circuit, hops=prune_hops)

        stim_types = self.circuit.input_types()
        # monitor_types is left empty on purpose: it adds a StateMonitor recording membrane
        # voltage for every cell of those types, which we never read. Spike counts come from
        # the SpikeMonitor CNSModel always installs, read as per-tick deltas in step().
        self.model = CNSModel(
            neurons, edges, stim_types, LIFParams(), electrical=True, monitor_types=()
        )
        self.n_neurons = int(len(self.model.lif_ids))
        self.n_edges = int(len(edges))
        self.missing_types = list(self.model.stim_missing)

        # --- stimulus rows: which channel and which side each Poisson source belongs to -----
        meta = neurons.set_index('bodyId')
        side = infer_side(meta)
        type_to_channel: dict[str, str] = {}
        for ch in INPUT_CHANNELS:
            for t in self.circuit.types(ch):
                type_to_channel.setdefault(t, ch)
        stim = self.model.stim
        row_ch = stim['type'].map(type_to_channel).fillna('')
        row_side = stim['bodyId'].map(side).where(lambda s: s.isin(['L', 'R']), 'L')
        self._rate_rows: dict[tuple[str, str], Any] = {}
        for ch in INPUT_CHANNELS:
            for s in ('L', 'R'):
                idx = np.flatnonzero(((row_ch == ch) & (row_side == s)).to_numpy())
                if len(idx):
                    self._rate_rows[(ch, s)] = idx
        self._rates = np.zeros(len(stim), dtype=float)

        # --- readout rows: G indices per output channel, split by side ----------------------
        self._read: dict[str, tuple[Any, Any]] = {}
        for ch in OUTPUT_CHANNELS:
            pos: list[int] = []
            for t in self.circuit.types(ch):
                pos += list(self.model.readout_index(t))
            pos = sorted(set(pos))
            if not pos:
                self._read[ch] = (np.array([], dtype=int), np.array([], dtype=int))
                continue
            ids = self.model.lif_ids[pos]
            s = side.reindex(ids).fillna('L').to_numpy()
            arr = np.asarray(pos, dtype=int)
            self._read[ch] = (arr[s == 'L'], arr[s != 'L'])

        self._last_count = np.array(self.model.spikes.count, dtype=np.int64).copy()
        self.ticks = 0
        self._rt: deque[float] = deque(maxlen=50)

    # --- the one hot call ----------------------------------------------------

    def step(
        self, drive: dict[str, tuple[float, float]], tick_ms: float | None = None
    ) -> BrainOut:
        """Drive the network for one tick and return the spikes its outputs produced."""
        np = self._np
        tick_ms = float(tick_ms or self.cfg.brain_tick_ms)
        self._rates.fill(0.0)
        for (ch, s), rows in self._rate_rows.items():
            left, right = drive.get(ch, (0.0, 0.0))
            self._rates[rows] = max(0.0, float(left if s == 'L' else right))
        self.model.set_stim_rates(self._rates)

        t0 = time.perf_counter()
        self.model.run(tick_ms)
        wall = (time.perf_counter() - t0) * 1000.0

        count = np.array(self.model.spikes.count, dtype=np.int64)
        delta = count - self._last_count
        self._last_count = count.copy()

        spikes, spikes_lr = {}, {}
        for ch, (li, ri) in self._read.items():
            nl = int(delta[li].sum()) if len(li) else 0
            nr = int(delta[ri].sum()) if len(ri) else 0
            spikes[ch] = nl + nr
            spikes_lr[ch] = (nl, nr)

        self.ticks += 1
        self._rt.append(tick_ms / max(wall, 1e-6))
        return BrainOut(tick_ms=tick_ms, spikes=spikes, spikes_lr=spikes_lr, wall_ms=wall)

    def status(self) -> BrainStatus:
        return BrainStatus(
            backend='flycns',
            loaded=True,
            n_neurons=self.n_neurons,
            n_edges=self.n_edges,
            n_stim=int(len(self.model.stim)),
            data_dir=self.data_dir,
            tick_ms=self.cfg.brain_tick_ms,
            realtime_factor=(sum(self._rt) / len(self._rt)) if self._rt else 0.0,
            missing_types=list(self.missing_types),
            ablated=list(self.ablated),
            ticks=self.ticks,
        )


# --- graph surgery -----------------------------------------------------------


def ablate_types(edges, neurons, types: list[str]):
    """Drop every edge touching a neuron of these types: the model analogue of a silencing."""
    ids = set(neurons.loc[neurons['type'].isin(types), 'bodyId'])
    if not ids:
        return edges
    keep = ~(edges['pre'].isin(ids) | edges['post'].isin(ids))
    return edges[keep].reset_index(drop=True)


def prune_to_circuit(neurons, edges, circuit: Circuit, hops: int = 3):
    """
    Keep only neurons within ``hops`` synapses of the circuit's own types.

    The full connectome is far slower than real time; a reflex only needs its own pathway.
    Always check a pruned result against the full graph before trusting it.
    """
    seeds = set(circuit.input_types()) | set(circuit.output_types())
    frontier = set(neurons.loc[neurons['type'].isin(seeds), 'bodyId'])
    keep = set(frontier)
    pre, post = edges['pre'].to_numpy(), edges['post'].to_numpy()
    for _ in range(max(0, hops)):
        if not frontier:
            break
        mask_f = [p in frontier for p in pre]
        fwd = set(post[[i for i, m in enumerate(mask_f) if m]])
        mask_b = [p in frontier for p in post]
        back = set(pre[[i for i, m in enumerate(mask_b) if m]])
        frontier = (fwd | back) - keep
        keep |= frontier
    n2 = neurons[neurons['bodyId'].isin(keep)].reset_index(drop=True)
    e2 = edges[edges['pre'].isin(keep) & edges['post'].isin(keep)].reset_index(drop=True)
    return n2, e2


def make_brain(
    backend: str | None = None,
    circuit: Circuit | None = None,
    cfg: Settings | None = None,
    ablate: list[str] | None = None,
    prune_hops: int | None = None,
):
    """Build whichever backend the environment asks for."""
    cfg = cfg or settings()
    backend = (backend or cfg.backend or 'mock').lower()
    if backend == 'flycns':
        return FlycnsBrain(circuit=circuit, cfg=cfg, ablate=ablate, prune_hops=prune_hops)
    if backend == 'mock':
        return MockBrain(circuit=circuit, cfg=cfg)
    raise ValueError(f'unknown BRAIN_BACKEND {backend!r}; choose mock or flycns')
