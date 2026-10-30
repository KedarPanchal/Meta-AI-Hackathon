"""The search-drone mission layer as an MCP server.

``FastMCP.from_fastapi(app=server)`` with ``server`` in ``app/main.py``: five tools, one prompt.
The brain (``fly_brain_pilot``) runs in its own process and is reached over ROS topics; with
``MISSION_LINK=sim`` the brain runs in-process over a kinematic world instead, which is how the
tests fly.

Run it::

    python mcp_server.py                # stdio, for an MCP client
    python mcp_server.py --http 8799    # HTTP, for poking at it with curl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
from fastmcp import FastMCP
from fastmcp.server.providers.openapi.routing import MCPType, RouteMap

from app.main import server

# A Muse call can take tens of seconds, and look_for_person makes one inline.
mcp = FastMCP.from_fastapi(
    app=server,
    name='flycns-search-drone',
    httpx_client_kwargs={'timeout': httpx.Timeout(300.0)},
    route_maps=[
        RouteMap(pattern=r'^/health$', mcp_type=MCPType.EXCLUDE),
        RouteMap(mcp_type=MCPType.TOOL),
    ],
)


@mcp.prompt
def find_person(area_hint: str = 'the whole map') -> str:
    """Run a search-and-rescue flight for a missing person and report where they are."""
    return f"""A person is missing in the forest, somewhere in {area_hint}. Find them with the drone.

Do this:

1. `mission_status`. Check `link` is connected and `detector` names a real model
   (`muse:...`). If the detector says `color`, Muse has no API key: say so before flying,
   because a colour blob is not a person detector.
2. `search_for_person` with `dry_run: true` first. Read the plan: lanes, path length,
   estimated duration, `coverage_estimate`, and any `warnings`. Narrow the `area` if you have a
   hint about where the person is; lower `altitude_m` (never below 12 while towers are in the
   area) if you need a bigger view of each person-sized object.
3. `search_for_person` for real. Then poll `mission_status` every 10-20 seconds and relay the
   `log` lines and `progress` to the operator. Do not call `abort_mission` because a lane looks
   slow: the brain may be side-stepping an obstacle, which shows up as `brain.state` AVOID or
   ESCAPE for a few seconds.
4. When `state` is `found`, report `best.estimated_xy` (world metres, x east, y north), the
   confidence, the description, and that the drone is holding above them. If the flight ends
   `complete` with no sighting, say which area was covered and suggest the next area or a lower
   altitude. If it is `failed`, read `note` and `log`: the usual causes are no odometry (the
   brain node is not running) or an unarmed vehicle.

Never report a position you did not get from `mission_status` or `look_for_person`."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--http', type=int, metavar='PORT', help='Serve over HTTP instead of stdio.')
    p.add_argument('--host', default='127.0.0.1', help='Bind address for --http.')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from mission.config import mission_settings

    cfg = mission_settings()
    if cfg.link in ('sim', 'auto') and cfg.brain.backend == 'flycns':
        # Only the in-process sim runs the brain here. Brian2 installs a SIGINT handler at
        # import, which Python only permits on the main thread; MCP handlers do not run there.
        from fly_brain.brain import preimport_flycns

        preimport_flycns(cfg.brain.repo)
    if args.http:
        mcp.run(transport='http', host=args.host, port=args.http)
    else:
        mcp.run()


if __name__ == '__main__':
    main()
