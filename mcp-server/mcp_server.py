"""FLYCNS drone brain as an MCP server.

Keeps the project's pattern -- ``FastMCP.from_fastapi(app=server)`` with ``server`` in
``app/main.py`` -- and adds the parts an agent needs beyond bare tools: two resources it can
read for context, and two prompts for the workflows that are easy to get wrong.

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

# A brain tick can take a long second on the full connectome, and test_reflex runs a hundred
# of them in one call, so the in-process ASGI client must not time out at httpx's default 5 s.
mcp = FastMCP.from_fastapi(
    app=server,
    name="flycns-drone",
    httpx_client_kwargs={"timeout": httpx.Timeout(900.0)},
    route_maps=[
        RouteMap(pattern=r"^/health$", mcp_type=MCPType.EXCLUDE),
        RouteMap(mcp_type=MCPType.TOOL),
    ],
)


@mcp.resource("flycns://docs/reflex", mime_type="text/markdown")
def reflex_doc() -> str:
    """The escape-reflex design note: why a giant-fibre spike is an event, not a rate."""
    path = Path(__file__).resolve().parent.parent / "REFLEX.md"
    if not path.exists():
        return "REFLEX.md is missing from the repository root."
    return path.read_text(encoding="utf-8")


@mcp.resource("flycns://circuit", mime_type="application/json")
def circuit_resource() -> dict:
    """The live channel-to-cell-type map, including which names the loaded graph actually has."""
    from fly_brain.controller import controller

    ctl = controller()
    with ctl.lock:
        return ctl.circuit.as_dict()


@mcp.prompt
def calibrate_reflex(clearance_m: float = 0.9, decel_mps2: float = 5.0) -> str:
    """Walk through tuning the escape reflex so it fires early enough to be useful."""
    required = 0.15 + (2.0 * clearance_m / decel_mps2) ** 0.5
    return f"""Calibrate the escape reflex on this aircraft.

The constraint: the drone needs {clearance_m} m of clearance and decelerates at
{decel_mps2} m/s^2, so with 0.15 s of sense-and-act latency the giant fibre has to fire at
least {required:.2f} s before contact.

Do this:

1. `brain_status`. Confirm the backend and read `realtime_factor`. Below 1.0 the brain is
   slower than the world; note it, because every latency number below is simulated time.
2. `get_motor_config` and note `encoder.gains.loom_speed`.
3. `test_reflex` with the object you actually care about, e.g. a 0.5 m object at 5 m/s.
   Read `ttc_at_trigger_s` and `verdict`.
4. If it fired too late, raise the loom gain with `update_motor_config`
   (`encoder_gains: {{"loom_speed": <g>, "loom_size": <g>}}`) and repeat 3. The response is
   monotonic but not linear; doubling the gain does not double the warning time.
5. If it never fired, check `get_circuit` first: `inputs.loom_speed` and `inputs.loom_size`
   must resolve to real types, and `outputs.gf` must resolve to DNp01.
6. Repeat at bearings of 0, +-30 and +-60 degrees. The eye split means a target off to one
   side drives one half of the population harder, so the trigger distance is not symmetric.
7. Re-run with `sound_db` at -20. On the real connectome FLYCNS reports sound LOWERS the
   giant-fibre loom response; on the toy graph it raises it. Record which way it goes on the
   dataset you are using, because the sign flips between them.
8. Keep the smallest gain that passes. A gain high enough to fire early also fires at
   harmless things, and the ground expands exactly like a loom while landing (see
   `reflex.suppress_below_horizon`).

Report the gain you chose, the trigger times at each bearing, and anything that fired when it
should not have."""


@mcp.prompt
def find_pursuit_neurons() -> str:
    """Find and calibrate the descending neurons that could steer the drone."""
    return """Wire up the pursuit decoder. `outputs.steer` and `outputs.forward` are empty, so
pursuit currently commands zero velocity and the drone just hovers.

The anatomy to respect: LC10a neurons track a small target and exist on both sides of the
head. A target on the right drives right LC10a harder. If some descending neuron inherits
that imbalance, its left-right difference is a steering signal. The P1/pC1 cells are NOT a
direction -- in FLYCNS they drive the song command neurons (pMP2, DNp13, pIP10), not
locomotion -- so they are only the gate.

Do this:

1. `find_neurons` with `^DN` to see what descending types this graph actually has. Do not
   guess names: MaleCNS splits populations into lettered subtypes and an unsplit name matches
   nothing at all.
2. `stimulate_neurons` on `["LC10a"]` at 20 Hz for 200 ms and read `most_active_types`. Then
   do the same with the arousal types from `get_circuit`. Candidates for steering are
   bilateral descending types that respond to LC10a; candidates for forward drive are types
   that respond to arousal.
3. `update_circuit` to set `steer` and `forward` to your candidates, then `load_brain`.
4. Verify it steers, do not assume it. `step_brain` with a target at `bearing_deg` of -60,
   -30, 0, +30, +60 and compare `spikes_lr.steer` across them. You want the left-right
   difference to change sign with the bearing. If it does not, the pathway does not carry
   direction and a different type is needed.
5. Only then fit the gains. `set_pursuit` with a `yaw_gain` from the imbalance you measured.
   Fit forward speed as a pure gain with NO intercept: an intercept makes the drone drift
   forward with the gate closed.
6. Check `set_pursuit` with `arousal: 0.0` still gives zero velocity.

Two things the model cannot do, so do not read them into the result: there is no
neuromodulation and no background activity, so inhibition-only and disinhibition effects
cannot appear, and P1 "gating" can only act as extra excitation summing downstream."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--http", type=int, metavar="PORT", help="Serve over HTTP instead of stdio.")
    p.add_argument("--host", default="127.0.0.1", help="Bind address for --http.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Brian2 installs a SIGINT handler at import, which Python only permits on the main
    # thread; MCP tool handlers do not run there. Import it here, while we still are.
    from fly_brain.brain import preimport_flycns
    from fly_brain.config import settings

    if settings().backend == "flycns":
        preimport_flycns(settings().repo)
    if args.http:
        mcp.run(transport="http", host=args.host, port=args.http)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
