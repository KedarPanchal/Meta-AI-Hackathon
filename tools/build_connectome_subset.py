#!/usr/bin/env python3
"""
Build the self-contained connectome subset used by ``fly_brain.connectome``.

The source is the public Janelia hemibrain v1.2.1 edge list and its cell-type
annotations.  The subset is the **induced one-hop subgraph** around the cell
types this project actually uses:

* visual loom inputs: LC4, LPLC2, LC10a
* giant-fibre/escape outputs: Giant Fiber, DNp02, DNp06, DNp11, DNg30,
  DNg79, DNg92_a
* reproductive pathway: pC1a..pC1e -> DNp13

"One hop" means every neuron with a chemical connection to or from one of
those seeds, plus every edge between the retained neurons.  That keeps the
real local wiring (including the many parallel LC4/LPLC2 contacts onto the
giant fibre) without shipping the 88 MB whole-brain table.  The output is a
gzipped JSON file with a provenance header, so the runtime has no
``flycns`` package, no Brian2, and no sibling checkout dependency.

This script is a build tool, not part of the runtime or the test loop.  Run it
only when refreshing the vendored data::

    pixi run python tools/build_connectome_subset.py

The source URLs are public and need no neuPrint token.  The dataset is
CC-BY 4.0; see the metadata in the generated file and the project README.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

SOURCE_BUCKET = 'https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/hemibrain_121'
META_NAME = 'hemibrain_121_meta.feather'
EDGE_NAME = 'hemibrain_121_simple_edgelist.feather'

# Exact cell types used by circuit.json.  Prefixes are deliberately not used
# here: the extractor is for the named cells, and a typo should fail loudly.
SEED_TYPES = (
    'LC4', 'LPLC2', 'LC10a',
    'Giant Fiber', 'DNp02', 'DNp06', 'DNp11', 'DNg30', 'DNg79', 'DNg92_a',
    'pC1a', 'pC1b', 'pC1c', 'pC1d', 'pC1e', 'DNp13',
)

# Same fly transmitter convention used by the source project: ACh excitatory;
# GABA, glutamate and histamine inhibitory; biogenic amines are modulatory and
# are kept in the metadata but contribute no sign in this model.
SIGN = {
    'acetylcholine': 1,
    'gaba': -1,
    'glutamate': -1,
    'histamine': -1,
    'dopamine': 0,
    'octopamine': 0,
    'serotonin': 0,
}


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as r, dest.open('wb') as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return dest


def side_of(instance: str) -> str:
    s = (instance or '').strip()
    m = re.search(r'(?:^|[_(])([LR])(?:$|[_)])', s)
    return m.group(1) if m else '?'


def build(meta_path: Path, edge_path: Path, out_path: Path) -> dict:
    import pandas as pd
    import pyarrow.feather as feather

    meta = feather.read_table(meta_path).to_pandas()
    meta['id'] = meta['hemibrain_121_id'].astype(str)
    meta['cell_type'] = meta['cell_type'].fillna('').astype(str)
    meta['nt'] = meta.get('neurotransmitter_predicted', '').fillna('').astype(str)

    missing = sorted(set(SEED_TYPES) - set(meta['cell_type']))
    if missing:
        raise SystemExit(f'source is missing seed types: {missing}')

    seeds = set(meta.loc[meta['cell_type'].isin(SEED_TYPES), 'id'])
    if not seeds:
        raise SystemExit('no seed neurons found')

    edges = feather.read_table(edge_path).to_pandas()
    edges['pre'] = edges['pre'].astype(str)
    edges['post'] = edges['post'].astype(str)

    # Both directions must be measured against the seeds, not against the
    # already-expanded set, or the second union silently turns this into a
    # two-hop (and then nearly whole-brain) subgraph.
    seed_frontier = set(seeds)
    keep = set(seeds)
    keep |= set(edges.loc[edges['pre'].isin(seed_frontier), 'post'])
    keep |= set(edges.loc[edges['post'].isin(seed_frontier), 'pre'])
    sub = edges[edges['pre'].isin(keep) & edges['post'].isin(keep)].copy()
    sub = sub[sub['count'] > 0]
    nodes = meta[meta['id'].isin(keep)].copy()

    # Stable order: cell type then body id.  A stable file makes diffs and tests
    # meaningful when the extractor is re-run.
    nodes = nodes.sort_values(['cell_type', 'id']).reset_index(drop=True)
    index = {body: i for i, body in enumerate(nodes['id'])}
    sub = sub[sub['pre'].isin(index) & sub['post'].isin(index)].copy()
    sub['pre_i'] = sub['pre'].map(index).astype(int)
    sub['post_i'] = sub['post'].map(index).astype(int)
    sub['sign'] = sub['pre'].map(nodes.set_index('id')['nt']).map(SIGN).fillna(0).astype(int)

    neurons = [
        {
            'id': str(row.id),
            'type': str(row.cell_type),
            'side': side_of(str(row.instance)),
            'nt': str(row.nt),
        }
        for row in nodes.itertuples(index=False)
    ]
    edge_list = [
        [int(row.pre_i), int(row.post_i), int(row.count), int(row.sign)]
        for row in sub.itertuples(index=False)
        if int(row.count) > 0
    ]
    payload = {
        'metadata': {
            'name': 'hemibrain_reflex_reproductive_subset',
            'source': 'hemibrain v1.2.1 (Janelia FlyEM / Cambridge Drosophila Connectomics)',
            'source_url': SOURCE_BUCKET,
            'license': 'CC-BY 4.0',
            'extracted': '2026-11-03',
            'method': (
                'induced one-hop chemical subgraph around ' + ', '.join(SEED_TYPES) +
                '; edges retain the EM synapse count and the presynaptic transmitter sign'
            ),
            'n_neurons': len(neurons),
            'n_edges': len(edge_list),
            'n_synapses': int(sum(e[2] for e in edge_list)),
            'seeds': list(SEED_TYPES),
            'notes': (
                'This is a real, local subset of a Drosophila connectome, not a toy graph '
                'and not a neural-activity recording. It is used as a spiking substrate; '
                'the only calibrated output is the escape trigger, and the pC1/DNp13 '
                'pathway is a commitment gate, not a heading decoder.'
            ),
        },
        'neurons': neurons,
        'edges': edge_list,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, 'wt', encoding='utf-8', compresslevel=9) as f:
        json.dump(payload, f, separators=(',', ':'))
    return payload['metadata']


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--meta', type=Path, help='local hemibrain metadata feather')
    ap.add_argument('--edges', type=Path, help='local hemibrain simple edge-list feather')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).resolve().parents[1] / 'src/fly_brain/fly_brain/data/hemibrain_subset.json.gz')
    ap.add_argument('--keep-downloads', action='store_true')
    args = ap.parse_args(argv)

    tmp = None
    try:
        if args.meta is None or args.edges is None:
            tmp = Path(tempfile.mkdtemp(prefix='flycns-subset-'))
        meta = args.meta or fetch(f'{SOURCE_BUCKET}/{META_NAME}', tmp / META_NAME)
        edges = args.edges or fetch(f'{SOURCE_BUCKET}/{EDGE_NAME}', tmp / EDGE_NAME)
        meta_info = build(meta, edges, args.out)
    finally:
        if tmp is not None and not args.keep_downloads:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(meta_info, indent=2))
    size = args.out.stat().st_size / 1e6
    print(f'wrote {args.out} ({size:.2f} MB)', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
