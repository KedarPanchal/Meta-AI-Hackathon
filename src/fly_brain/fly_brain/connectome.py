"""
A self-contained spiking runtime for the vendored *Drosophila* hemibrain subset.

There is no ``flycns`` package, no Brian2 import and no sibling checkout here.  The
data file is an induced one-hop subgraph of the public hemibrain v1.2.1
connectome, built by :mod:`tools.build_connectome_subset`: 5,573 neurons and
712,973 signed chemical edges around the loom/escape and pC1/DNp13 pathways.

The simulator is deliberately small and explicit: leaky integrate-and-fire
neurons, Poisson sensory drive, a sparse CSR synapse matrix, and a 1 ms
integration step inside each brain tick.  It is a control substrate, not a
claim to reproduce every membrane property of the fly.  The two properties the
motor layer depends on are preserved:

* a giant-fibre spike is an **event** (one or two spikes to a strong loom), not a
  rate proportional to the stimulus; and
* the pC1 -> DNp13 pathway is a **commitment gate**, not a heading decoder.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import gzip
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .circuit import Circuit, INPUT_CHANNELS, OUTPUT_CHANNELS
from .config import Settings, settings
from .schemas import BrainOut, BrainStatus

DEFAULT_DATA = Path(__file__).with_name('data') / 'hemibrain_subset.json.gz'

# The hemibrain subset is signed by presynaptic transmitter.  These are the same
# conventions as the source graph: ACh is excitatory; GABA, glutamate and
# histamine are inhibitory; amines are neuromodulatory and have no fast current.
SIGN_MAP = {
    'acetylcholine': 1.0,
    'gaba': -1.0,
    'glutamate': -1.0,
    'histamine': -1.0,
    'dopamine': 0.0,
    'octopamine': 0.0,
    'serotonin': 0.0,
}


@dataclass
class LIFParams:
    """Parameters of the local LIF substrate, in ms and mV."""

    dt_ms: float = 1.0
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    v_rest_mv: float = -60.0
    v_thresh_mv: float = -50.0
    v_reset_mv: float = -62.0
    refractory_ms: float = 2.0
    syn_gain_mv: float = 16.0
    """mV added to the postsynaptic current per 1,000 EM synapses in one spike.

    Calibrated against the encoder on the vendored subset: a 40 deg / 200 dps loom
    fires the giant fibre, while a 12 deg / 3 dps loom and empty sky do not.  The
    value is a single global scale; per-synapse weights are the real EM counts.
    """


@dataclass
class ConnectomeSubset:
    """The vendored graph plus lookup tables by cell type and side."""

    neurons: list[dict[str, Any]]
    edges: list[list[int]]
    metadata: dict[str, Any]
    path: Path
    type_ids: dict[str, list[int]] = field(default_factory=dict)
    side_of: np.ndarray | None = None
    sign_of: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.type_ids = {}
        for i, n in enumerate(self.neurons):
            self.type_ids.setdefault(str(n.get('type', '')), []).append(i)
        self.side_of = np.asarray([str(n.get('side', '?')) for n in self.neurons])
        self.sign_of = np.asarray(
            [SIGN_MAP.get(str(n.get('nt', '')).lower(), 0.0) for n in self.neurons],
            dtype=np.float32,
        )

    @classmethod
    def load(cls, path: Path | str | None = None) -> 'ConnectomeSubset':
        p = Path(path) if path else DEFAULT_DATA
        if not p.exists():
            raise RuntimeError(
                f'connectome subset not found at {p}. It is vendored with the repository; '
                'rebuild it with `pixi run python tools/build_connectome_subset.py` if it was removed.'
            )
        opener = gzip.open if p.suffix == '.gz' else open
        with opener(p, 'rt', encoding='utf-8') as f:
            raw = json.load(f)
        return cls(
            neurons=list(raw['neurons']), edges=[list(e) for e in raw['edges']],
            metadata=dict(raw.get('metadata', {})), path=p,
        )

    def ids_for(self, cell_type: str) -> list[int]:
        return list(self.type_ids.get(cell_type, []))


class LIFNetwork:
    """Sparse LIF network with Poisson input rows and per-cell-type readout."""

    def __init__(self, subset: ConnectomeSubset, params: LIFParams | None = None,
                 seed: int = 7):
        self.subset = subset
        self.params = params or LIFParams()
        self.rng = np.random.default_rng(seed)
        self.n = len(subset.neurons)
        pre = np.asarray([e[0] for e in subset.edges], dtype=np.int32)
        post = np.asarray([e[1] for e in subset.edges], dtype=np.int32)
        count = np.asarray([e[2] for e in subset.edges], dtype=np.float32)
        sign = np.asarray([e[3] for e in subset.edges], dtype=np.float32)
        # A zero sign is a modulatory transmitter or an unknown one: no fast
        # current.  Dropping it here keeps the spike loop cheap.
        keep = (sign != 0.0) & (count > 0)
        pre, post = pre[keep], post[keep]
        self.w = (count[keep] / 1000.0) * sign[keep] * self.params.syn_gain_mv
        order = np.argsort(pre, kind='stable')
        self.pre = pre[order]
        self.post = post[order]
        self.w = self.w[order]
        self.indptr = np.zeros(self.n + 1, dtype=np.int64)
        np.add.at(self.indptr, self.pre + 1, 1)
        self.indptr = np.cumsum(self.indptr)
        self.v = np.full(self.n, self.params.v_rest_mv, dtype=np.float32)
        self.i_syn = np.zeros(self.n, dtype=np.float32)
        self.ref_until = np.zeros(self.n, dtype=np.int32)
        self.step_i = 0

    def _spike_edges(self, active: np.ndarray) -> None:
        """Add the current from every outgoing edge of the active cells."""
        if active.size == 0:
            return
        starts = self.indptr[active]
        stops = self.indptr[active + 1]
        lengths = stops - starts
        total = int(lengths.sum())
        if total == 0:
            return
        # Gather the edge slices without a Python loop over active cells.
        offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths)
        idx = np.arange(total, dtype=np.int64) + offsets
        np.add.at(self.i_syn, self.post[idx], self.w[idx])

    def run(self, drive: dict[str, tuple[float, float]], tick_ms: float,
            input_ids: dict[str, tuple[np.ndarray, np.ndarray]],
            output_ids: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
        """Advance one tick; return total and left/right spike counts per output."""
        p = self.params
        n_steps = max(1, int(round(float(tick_ms) / p.dt_ms)))
        counts = np.zeros(self.n, dtype=np.int32)
        # Injected sensory spikes are real events on the stimulated cells.  A
        # readout may legitimately include one of them (pC1 is both the arousal
        # input and the reproductive readout), so count them too.
        injected = np.zeros(self.n, dtype=np.int32)
        secs = p.dt_ms / 1000.0
        for _ in range(n_steps):
            self.step_i += 1
            self.i_syn *= float(np.exp(-p.dt_ms / p.tau_syn_ms))
            # Sensory Poisson drive.  The sensory cells still exist in the
            # network; this injects their spikes rather than replacing them.
            for ch, (li, ri) in input_ids.items():
                left, right = drive.get(ch, (0.0, 0.0))
                if li.size:
                    hit = self.rng.random(li.size) < max(0.0, left) * secs
                    if hit.any():
                        injected[li[hit]] += 1
                        self._spike_edges(li[hit])
                if ri.size:
                    hit = self.rng.random(ri.size) < max(0.0, right) * secs
                    if hit.any():
                        injected[ri[hit]] += 1
                        self._spike_edges(ri[hit])

            eligible = (self.v >= p.v_thresh_mv) & (self.ref_until <= self.step_i)
            active = np.flatnonzero(eligible)
            if active.size:
                counts[active] += 1
                self.v[active] = p.v_reset_mv
                self.ref_until[active] = self.step_i + int(round(p.refractory_ms / p.dt_ms))
                self._spike_edges(active)

            self.v += (p.dt_ms / p.tau_m_ms) * (-(self.v - p.v_rest_mv) + self.i_syn)
            self.v[self.ref_until > self.step_i] = p.v_reset_mv

        all_counts = counts + injected
        total: dict[str, int] = {}
        lr: dict[str, tuple[int, int]] = {}
        for ch, (li, ri) in output_ids.items():
            nl = int(all_counts[li].sum()) if li.size else 0
            nr = int(all_counts[ri].sum()) if ri.size else 0
            total[ch] = nl + nr
            lr[ch] = (nl, nr)
        return total, lr


class ConnectomeBrain:
    """The real subset as a closed-loop controller: drive in, spikes out."""

    backend = 'connectome'

    def __init__(self, circuit: Circuit | None = None, cfg: Settings | None = None,
                 ablate: list[str] | None = None, prune_hops: int | None = None):
        # ``prune_hops`` and ``ablate`` are kept in the constructor because the
        # controller and older callers pass them.  The vendored subset is already
        # pruned; an explicit ablation is applied to the circuit lookup, not to
        # the data file, so a reload is deterministic.
        del prune_hops
        self.cfg = cfg or settings()
        self.subset = ConnectomeSubset.load(self.cfg.connectome_path)
        self.circuit = circuit or Circuit()
        self.circuit.resolve(self.subset.type_ids.keys())
        self.ablated = list(ablate or [])
        self.net = LIFNetwork(self.subset)
        self.n_neurons = self.net.n
        self.n_edges = int(len(self.subset.edges))
        self.data_dir = str(self.subset.path)
        self.ticks = 0
        self._rt: deque[float] = deque(maxlen=50)
        self._input_ids = self._build_inputs()
        self._output_ids = self._build_outputs()

    def _build_inputs(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for ch in INPUT_CHANNELS:
            ids: list[int] = []
            for t in self.circuit.types(ch):
                ids += self.subset.ids_for(t)
            ids = sorted(set(ids))
            if not ids:
                out[ch] = (np.array([], dtype=int), np.array([], dtype=int))
                continue
            arr = np.asarray(ids, dtype=int)
            side = self.subset.side_of[arr]
            out[ch] = (arr[side != 'R'], arr[side == 'R'])
        return out

    def _build_outputs(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for ch in OUTPUT_CHANNELS:
            ids: list[int] = []
            for t in self.circuit.types(ch):
                if t in self.ablated:
                    continue
                ids += self.subset.ids_for(t)
            ids = sorted(set(ids))
            if not ids:
                out[ch] = (np.array([], dtype=int), np.array([], dtype=int))
                continue
            arr = np.asarray(ids, dtype=int)
            side = self.subset.side_of[arr]
            out[ch] = (arr[side != 'R'], arr[side == 'R'])
        return out

    def step(self, drive: dict[str, tuple[float, float]],
             tick_ms: float | None = None) -> BrainOut:
        tick_ms = float(tick_ms or self.cfg.brain_tick_ms)
        t0 = time.perf_counter()
        spikes, spikes_lr = self.net.run(drive, tick_ms, self._input_ids, self._output_ids)
        wall = (time.perf_counter() - t0) * 1000.0
        self.ticks += 1
        self._rt.append(tick_ms / max(wall, 1e-6))
        return BrainOut(tick_ms=tick_ms, spikes=spikes, spikes_lr=spikes_lr, wall_ms=wall)

    def status(self) -> BrainStatus:
        return BrainStatus(
            backend=self.backend,
            loaded=True,
            n_neurons=self.n_neurons,
            n_edges=self.n_edges,
            n_stim=sum(len(a) + len(b) for a, b in self._input_ids.values()),
            data_dir=self.data_dir,
            tick_ms=self.cfg.brain_tick_ms,
            realtime_factor=(sum(self._rt) / len(self._rt)) if self._rt else 0.0,
            missing_types=[t for ts in self.circuit.missing.values() for t in ts],
            ablated=list(self.ablated),
            ticks=self.ticks,
        )

    def stimulate(self, types: list[str], hz: float = 20.0, ms: float = 200.0) -> dict:
        """Run a separate copy for an experiment; never disturb the flying network."""
        present = set(self.subset.type_ids)
        found = [t for t in types if t in present]
        if not found:
            return {'error': f'none of {types} exist in {self.data_dir}', 'spikes': {}}
        circuit = Circuit()
        for ch in INPUT_CHANNELS:
            circuit.patch(ch, [])
        for t in found:
            circuit.patch('arousal', found)
        brain = ConnectomeBrain(circuit=circuit, cfg=self.cfg)
        rates = {ch: (hz, hz) for ch in ('loom_speed', 'loom_size', 'target', 'sound', 'arousal')}
        before = brain.step({ch: (0.0, 0.0) for ch in rates}, 1.0)
        out = brain.step(rates, ms)
        return {
            'backend': self.backend, 'stimulated': found, 'hz': hz, 'ms': ms,
            'spikes': out.spikes, 'control': before.spikes,
            'note': 'independent copy; the flying network is untouched',
        }
