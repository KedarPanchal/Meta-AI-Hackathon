"""Inspecting and configuring the brain. Seven tools."""
from __future__ import annotations

import re
from typing import Annotated, Literal

from fastapi import APIRouter, Body, Query
from pydantic import BaseModel, Field

from fly_brain.brain import preimport_flycns
from fly_brain.controller import controller
from fly_brain.schemas import BrainStatus, Percept

router = APIRouter(prefix="/brain", tags=["brain"])


class LoadRequest(BaseModel):
    backend: Literal["mock", "flycns"] | None = Field(
        default=None, description="Which brain to build. Defaults to BRAIN_BACKEND."
    )
    ablate: list[str] = Field(
        default_factory=list,
        description="Cell types to silence: every edge touching them is dropped before the "
        "network is built. The model analogue of a silencing experiment.",
    )
    prune_hops: int | None = Field(
        default=None,
        description="Keep only neurons within this many synapses of the circuit's own types. "
        "The full connectome is far slower than real time; a reflex only needs its pathway. "
        "Always compare a pruned result against the full graph.",
    )


class StepRequest(BaseModel):
    percept: Percept = Field(default_factory=Percept, description="What the brain is told this tick.")
    ticks: int = Field(default=1, ge=1, le=200, description="How many ticks to run.")
    send: bool = Field(
        default=False,
        description="Whether the resulting command may reach the vehicle. False by default: "
        "stepping the brain is an inspection, not a flight command.",
    )


class CircuitPatch(BaseModel):
    channel: str = Field(description="Channel to change, e.g. 'steer' or 'loom_speed'.")
    types: list[str] = Field(
        description="Cell types for that channel. Matched by EXACT name against the loaded "
        "graph; a trailing '*' expands by prefix, which is how you catch MaleCNS's split "
        "populations (DNg02 does not exist, DNg02_a..g do)."
    )
    save: bool = Field(default=False, description="Also write the change back to circuit.json.")


class StimulateRequest(BaseModel):
    types: list[str] = Field(description="Cell types to drive, by exact name.")
    hz: float = Field(default=20.0, gt=0, le=500, description="Poisson rate per cell, Hz.")
    ms: float = Field(default=200.0, gt=0, le=5000, description="How long to drive them, ms.")


@router.get("/status", operation_id="brain_status", response_model=BrainStatus)
def brain_status() -> BrainStatus:
    """What brain is loaded, how big it is, and whether it keeps up with real time.

    ``realtime_factor`` is simulated time over wall time. Below 1.0 the brain is slower than
    the world, so either raise BRAIN_TICK_MS, prune the graph, or lower the simulator's
    real-time factor to match.
    """
    return controller().status()


@router.post("/load", operation_id="load_brain", response_model=BrainStatus)
def load_brain(req: Annotated[LoadRequest, Body()] = LoadRequest()) -> BrainStatus:
    """Build or rebuild the network, optionally ablated or pruned.

    This replaces the brain the flight loop is using, so it resets the escape clock.
    """
    req = req or LoadRequest()
    return controller().load_brain(
        backend=req.backend, ablate=req.ablate, prune_hops=req.prune_hops
    )


@router.get("/neurons", operation_id="find_neurons")
def find_neurons(
    pattern: Annotated[str, Query(description="Case-insensitive regex over cell type names.")],
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> dict:
    """Search the graph's cell types by regex, with a count of cells for each.

    Use this before naming a type anywhere else. Stimulus and readout types are matched by
    exact name, and MaleCNS splits many populations into lettered subtypes, so a plausible
    name often matches nothing at all.
    """
    ctl = controller()
    cfg = ctl.cfg
    if not preimport_flycns(cfg.repo):
        return {"error": "flycns is not importable; set FLYDRONE_REPO to the clone.", "types": {}}
    from flycns import load_graph

    neurons, _ = load_graph(str(cfg.flycns_data_dir()))
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return {"error": f"bad regex: {exc}", "types": {}}
    counts = neurons["type"].dropna().value_counts()
    hits = {str(t): int(n) for t, n in counts.items() if rx.search(str(t))}
    ordered = dict(sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))[:limit])
    return {
        "pattern": pattern,
        "data_dir": str(cfg.flycns_data_dir()),
        "n_matching_types": len(hits),
        "types": ordered,
        "truncated": len(hits) > len(ordered),
    }


@router.get("/circuit", operation_id="get_circuit")
def get_circuit() -> dict:
    """The channel-to-cell-type map, with which names were actually found in the graph.

    ``resolved`` is what is driving or being read; ``missing`` is what was asked for and is
    not there. ``outputs.steer`` and ``outputs.forward`` start empty: the pursuit decoder is
    not wired until they are filled and calibrated.
    """
    ctl = controller()
    with ctl.lock:
        return ctl.circuit.as_dict()


@router.patch("/circuit", operation_id="update_circuit")
def update_circuit(patch: Annotated[CircuitPatch, Body()]) -> dict:
    """Point a channel at different cell types.

    Takes effect on the next ``load_brain``: stimulus and readout wiring is fixed when the
    network is built.
    """
    ctl = controller()
    with ctl.lock:
        try:
            new = ctl.circuit.patch(patch.channel, patch.types)
        except KeyError as exc:
            return {"error": str(exc)}
        saved = str(ctl.circuit.save()) if patch.save else None
        return {
            "channel": patch.channel,
            "types": new,
            "saved_to": saved,
            "note": "call load_brain to rebuild the network with this wiring.",
        }


@router.post("/step", operation_id="step_brain")
def step_brain(req: Annotated[StepRequest, Body()] = StepRequest()) -> dict:
    """Run the brain on a percept you supply and report the spikes and the command.

    The main experiment tool: hold a target at a bearing, see which descending neurons fire
    and what velocity the motor layer would ask for.
    """
    req = req or StepRequest()
    ctl = controller()
    rows = []
    for _ in range(req.ticks):
        cmd = ctl.tick(req.percept, send=req.send)
        out = ctl.last_out
        rows.append(
            {
                "spikes": out.spikes if out else {},
                "spikes_lr": out.spikes_lr if out else {},
                "wall_ms": round(out.wall_ms, 2) if out else 0.0,
                "command": cmd.model_dump(),
            }
        )
    totals: dict[str, int] = {}
    for r in rows:
        for k, v in r["spikes"].items():
            totals[k] = totals.get(k, 0) + int(v)
    return {"ticks": req.ticks, "totals": totals, "steps": rows, "status": ctl.status().model_dump()}


@router.post("/stimulate", operation_id="stimulate_neurons")
def stimulate_neurons(req: Annotated[StimulateRequest, Body()]) -> dict:
    """Drive arbitrary cell types on a throwaway copy of the network and see what follows.

    A separate copy on purpose: CNSModel replaces stimulated types with Poisson sources, so
    doing this on the live brain would delete those cells from the flight controller's network
    and they could never be read back.
    """
    return controller().stimulate(req.types, hz=req.hz, ms=req.ms)
