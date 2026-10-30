"""The FastAPI application. ``mcp_server.py`` turns this into an MCP server.

Every route's ``operation_id`` is the MCP tool name, so they are chosen to read as tool names
rather than as URLs. ``/health`` is the one route excluded from the tool list.

There are no startup or lifespan events here, and there must not be: ``FastMCP.from_fastapi``
calls the app in-process through an httpx ASGI transport, which never runs them. The brain
link and the detector are built on first use (``mission.runner.MissionService``).
"""
from __future__ import annotations

from .bootstrap import ensure_fly_brain

FLY_BRAIN_SOURCE = ensure_fly_brain()

from fastapi import FastAPI  # noqa: E402

from mission.config import mission_settings  # noqa: E402

from .routers import mission  # noqa: E402

server = FastAPI(
    title='FLYCNS search drone: mission layer',
    version='0.2.0',
    summary='Find a person in the forest, or scout all of it, with a drone flown by a fly brain.',
    description=(
        'This server is the MISSION layer. It decides WHERE the drone should go and WHAT the '
        'camera shows; it never commands velocity. Flying, obstacle avoidance and the escape '
        'reflex belong to the brain (`fly_brain`, a connectome-constrained simulation of the '
        'fly nervous system), which receives one goal position at a time on `/fly_brain/goal`.\n\n'
        '* `search_for_person` plans a boustrophedon coverage path over the area, flies it one '
        'waypoint at a time, runs Meta\'s Muse Spark vision model over the downward camera, '
        'confirms a sighting from directly above, and ends hovering over the person.\n'
        '* `scout_area` flies the same coverage and reports everyone seen.\n'
        '* `mission_status`, `abort_mission`, `look_for_person` watch, stop, and check once.\n\n'
        'Coordinates are world ENU metres: x east, y north, z up. The map is 120 x 120 m centred '
        'on the origin; the forest floor spans about -2.5..3 m in z. A goal arms the drone.'
    ),
)

server.include_router(mission.router)


@server.get('/health', operation_id='health', include_in_schema=True, tags=['meta'])
def health() -> dict:
    """Liveness and configuration. Not exposed as a tool."""
    cfg = mission_settings()
    return {
        'ok': True,
        'fly_brain': FLY_BRAIN_SOURCE,
        'link': cfg.link,
        'detector': cfg.detector,
        'muse_model': cfg.muse_model,
        'muse_key_set': bool(cfg.muse_api_key),
        'altitude_m': cfg.altitude_m,
        'topics': {
            'goal': cfg.brain.topic_goal,
            'goal_status': cfg.brain.topic_goal_status,
            'odom': cfg.brain.topic_odom,
            'image_down': cfg.brain.topic_image_down,
        },
    }
