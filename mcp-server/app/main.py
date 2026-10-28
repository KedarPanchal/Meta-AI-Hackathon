"""The FastAPI application. ``mcp_server.py`` turns this into an MCP server.

Every route's ``operation_id`` is the MCP tool name, so they are chosen to read as tool names
rather than as URLs. ``/health`` is the one route excluded from the tool list.

There are no startup or lifespan events here, and there must not be: ``FastMCP.from_fastapi``
calls the app in-process through an httpx ASGI transport, which never runs them. Anything
that needs setting up is set up on first use (``Controller.ensure_brain``).
"""
from __future__ import annotations

from .bootstrap import ensure_fly_brain

FLY_BRAIN_SOURCE = ensure_fly_brain()

from fastapi import FastAPI  # noqa: E402

from fly_brain.config import settings  # noqa: E402

from .routers import brain, drone, reflex  # noqa: E402

server = FastAPI(
    title="FLYCNS drone brain",
    version="0.1.0",
    summary="Fly a multirotor with a connectome-constrained simulation of a fly's nervous system.",
    description=(
        "The drone's camera and microphone drive sensory neurons of the FLYCNS MaleCNS model; "
        "spikes from descending and motor neurons become velocity.\n\n"
        "Two behaviours:\n\n"
        "* **Escape** -- the giant fibre (DNp01) fires one or two spikes to a looming object. "
        "That spike is a trigger EVENT, not a rate, so it commits the drone to a fixed "
        "manoeuvre away from the threat rather than driving a feedback loop.\n"
        "* **Pursuit** -- LC10a target neurons, split left and right, steer toward a target "
        "while the P1/pC1 courtship cells act as a 'pursue now' gate. The gate holds a state, "
        "not a direction; the direction comes from vision.\n\n"
        "Conventions: body frame FLU (x forward, y left, z up); yaw_rate positive is left; "
        "bearing_deg positive is RIGHT of the nose.\n\n"
        "Output is disarmed by default. Nothing reaches the vehicle until `enable_output`."
    ),
)

server.include_router(brain.router)
server.include_router(reflex.router)
server.include_router(drone.router)


@server.get("/health", operation_id="health", include_in_schema=True, tags=["meta"])
def health() -> dict:
    """Liveness and how the server found the brain package. Not exposed as a tool."""
    cfg = settings()
    return {
        "ok": True,
        "fly_brain": FLY_BRAIN_SOURCE,
        "backend": cfg.backend,
        "repo": str(cfg.repo),
        "data_dir": str(cfg.flycns_data_dir()),
        "tick_ms": cfg.brain_tick_ms,
    }
