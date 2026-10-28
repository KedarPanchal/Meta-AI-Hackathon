"""
The channel -> cell-type map, and resolving it against a loaded graph.

FLYCNS matches stimulus types by exact name and MaleCNS splits many populations into
subtypes, so a name that looks right can silently drive nothing. ``Circuit.resolve()``
checks every candidate against the graph that is actually loaded and reports what is
missing instead of failing quietly.
"""
from __future__ import annotations

from collections.abc import Iterable
import copy
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name('circuit.json')

INPUT_CHANNELS = ('loom_speed', 'loom_size', 'target', 'sound', 'arousal')
OUTPUT_CHANNELS = ('gf', 'jump', 'flight_power', 'steer', 'forward')


class Circuit:
    """Mutable channel -> types map, loaded from JSON and patchable at runtime."""

    def __init__(self, spec: dict | None = None, path: Path | str | None = None):
        if spec is None:
            p = Path(path) if path else DEFAULT_PATH
            spec = json.loads(Path(p).read_text(encoding='utf-8'))
            self.path = Path(p)
        else:
            self.path = Path(path) if path else DEFAULT_PATH
        self.spec = spec
        # types actually found in the graph, filled by resolve()
        self.resolved: dict[str, list[str]] = {}
        self.missing: dict[str, list[str]] = {}

    # --- reading -------------------------------------------------------------

    def candidates(self, channel: str) -> list[str]:
        """Return the declared type names (a trailing ``*`` is a prefix) for one channel."""
        for group in ('inputs', 'outputs'):
            if channel in self.spec.get(group, {}):
                return list(self.spec[group][channel].get('types', []))
        raise KeyError(f'unknown channel {channel!r}')

    def note(self, channel: str) -> str:
        for group in ('inputs', 'outputs'):
            if channel in self.spec.get(group, {}):
                return str(self.spec[group][channel].get('note', ''))
        return ''

    def types(self, channel: str) -> list[str]:
        """Return the resolved types if ``resolve()`` has run, else the raw candidates."""
        return self.resolved.get(channel, self.candidates(channel))

    def input_types(self) -> list[str]:
        """Every resolved input type, de-duplicated, in channel order."""
        return _dedup(t for c in INPUT_CHANNELS for t in self.types(c))

    def output_types(self) -> list[str]:
        return _dedup(t for c in OUTPUT_CHANNELS for t in self.types(c))

    # --- resolving -----------------------------------------------------------

    def resolve(self, available: Iterable[str]) -> dict[str, list[str]]:
        """
        Keep the candidates present in ``available``; expand ``Name*`` by prefix.

        Records what was dropped in ``self.missing`` so the caller can say so out loud.
        """
        have = {str(t) for t in available if t is not None and str(t) != 'nan'}
        self.resolved, self.missing = {}, {}
        for channel in INPUT_CHANNELS + OUTPUT_CHANNELS:
            found: list[str] = []
            gone: list[str] = []
            for cand in self.candidates(channel):
                if cand.endswith('*'):
                    hits = sorted(t for t in have if t.startswith(cand[:-1]))
                    found += hits
                    if not hits:
                        gone.append(cand)
                elif cand in have:
                    found.append(cand)
                else:
                    gone.append(cand)
            self.resolved[channel] = _dedup(found)
            self.missing[channel] = gone
        return self.resolved

    # --- writing -------------------------------------------------------------

    def patch(self, channel: str, types: list[str]) -> list[str]:
        """Replace one channel's type list. Returns the new list."""
        for group in ('inputs', 'outputs'):
            if channel in self.spec.get(group, {}):
                self.spec[group][channel]['types'] = list(types)
                self.resolved.pop(channel, None)
                self.missing.pop(channel, None)
                return list(types)
        raise KeyError(f'unknown channel {channel!r}')

    def as_dict(self) -> dict:
        """JSON-safe view including what resolved, for the MCP ``get_circuit`` tool."""
        out = copy.deepcopy(self.spec)
        out.pop('_comment', None)
        for group in ('inputs', 'outputs'):
            for channel, entry in out.get(group, {}).items():
                if channel in self.resolved:
                    entry['resolved'] = self.resolved[channel]
                    entry['missing'] = self.missing.get(channel, [])
        out['path'] = str(self.path)
        return out

    def save(self, path: Path | str | None = None) -> Path:
        p = Path(path) if path else self.path
        p.write_text(json.dumps(self.spec, indent=2) + '\n', encoding='utf-8')
        return p


def _dedup(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))
