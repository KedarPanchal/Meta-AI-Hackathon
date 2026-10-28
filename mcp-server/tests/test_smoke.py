"""Smoke tests for the FLYCNS drone brain, over both faces of the server.

Runs against whatever ``BRAIN_BACKEND`` says, so the same file covers the mock brain and the
real spiking network::

    python tests/test_smoke.py
    BRAIN_BACKEND=flycns FLYDRONE_REPO=/path/to/FLYCNS FLYDRONE_DATA=testdata python tests/test_smoke.py

The flycns-only checks skip themselves on the mock backend rather than failing, so one
command is honest in both configurations.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from fastmcp import Client  # noqa: E402

from app.main import server  # noqa: E402
from mcp_server import mcp  # noqa: E402

BACKEND = os.environ.get("BRAIN_BACKEND", "mock")
IS_FLYCNS = BACKEND == "flycns"

if IS_FLYCNS:
    # Import Brian2 here, at module scope, because this is the main thread. FastAPI's
    # TestClient runs request handlers on a worker thread, and Brian2 installs a SIGINT
    # handler when it is imported, which Python only permits on the main thread. Without this
    # the first lazy brain load inside a request fails with "signal only works in main
    # thread" -- which is what the real entry points (mcp_server.main, fly_drone.node) avoid
    # by doing exactly this at startup.
    from fly_brain.brain import flycns_import_error, preimport_flycns
    from fly_brain.config import settings

    if not preimport_flycns(settings().repo):
        raise SystemExit(
            f"BRAIN_BACKEND=flycns but flycns is not importable: {flycns_import_error()}"
        )

EXPECTED_TOOLS = {
    "brain_status", "load_brain", "find_neurons", "get_circuit", "update_circuit",
    "step_brain", "stimulate_neurons", "get_motor_config", "update_motor_config",
    "reflex_events", "test_reflex", "drone_state", "enable_output", "set_pursuit",
    "stop_drone",
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(server)


def test_health(client: TestClient) -> None:
    """The one route that is not a tool answers, and says how it found the brain package."""
    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["backend"] == BACKEND


def test_mcp_surface() -> None:
    """Exactly the fifteen tools, both resources and both prompts, and /health excluded."""

    async def go() -> None:
        async with Client(mcp) as c:
            tools = {t.name for t in await c.list_tools()}
            assert tools == EXPECTED_TOOLS, f"unexpected tool set: {tools ^ EXPECTED_TOOLS}"
            assert "health" not in tools
            uris = {str(r.uri) for r in await c.list_resources()}
            assert uris == {"flycns://docs/reflex", "flycns://circuit"}
            prompts = {p.name for p in await c.list_prompts()}
            assert prompts == {"calibrate_reflex", "find_pursuit_neurons"}

    asyncio.run(go())


def test_brain_loads_and_reports(client: TestClient) -> None:
    """The brain builds lazily and then reports a real size and speed."""
    r = client.post("/brain/load", json={})
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["loaded"] is True
    assert st["backend"] == BACKEND
    if IS_FLYCNS:
        assert st["n_neurons"] > 0
        assert st["n_edges"] > 0
        assert st["n_stim"] > 0


def test_circuit_resolves_against_the_graph(client: TestClient) -> None:
    """Every input channel finds at least one real cell type; steer/forward start empty."""
    r = client.get("/brain/circuit")
    assert r.status_code == 200, r.text
    c = r.json()
    for channel in ("loom_speed", "loom_size", "target", "sound", "arousal"):
        assert c["inputs"][channel]["resolved"], f"{channel} resolved to nothing"
    assert c["outputs"]["gf"]["resolved"] == ["DNp01"]
    assert c["outputs"]["steer"]["resolved"] == []


def test_circuit_patch_round_trip(client: TestClient) -> None:
    """A channel can be repointed, and the response says a reload is needed."""
    r = client.patch("/brain/circuit", json={"channel": "steer", "types": ["DNa02"], "save": False})
    assert r.status_code == 200, r.text
    assert r.json()["types"] == ["DNa02"]
    assert "load_brain" in r.json()["note"]
    r = client.patch("/brain/circuit", json={"channel": "steer", "types": [], "save": False})
    assert r.json()["types"] == []
    assert client.patch(
        "/brain/circuit", json={"channel": "nonsense", "types": []}
    ).json().get("error")


def test_step_brain_does_not_move_the_vehicle(client: TestClient) -> None:
    """Stepping is an inspection: it computes a command but output stays shut."""
    percept = {
        "has_target": True, "bearing_deg": 0.0, "size_deg": 40.0,
        "expansion_dps": 350.0, "altitude_m": 3.0,
    }
    r = client.post("/brain/step", json={"percept": percept, "ticks": 2, "send": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticks"] == 2
    assert len(body["steps"]) == 2
    assert "gf" in body["totals"]
    assert client.get("/drone/state").json()["output_enabled"] is False


def test_escape_fires_and_runs_away(client: TestClient) -> None:
    """A real loom triggers ESCAPE, and the escape points away from the threat and up."""
    client.post("/brain/load", json={})
    r = client.post(
        "/reflex/test",
        json={"distance_m": 5.0, "speed_mps": 5.0, "object_size_m": 0.5, "bearing_deg": 0.0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trigger"] is not None, f"reflex never fired: {body['verdict']}"
    cmd = body["trigger"]["command"]
    assert cmd["state"] == "ESCAPE"
    assert cmd["vx"] < 0, "escape should back away from a head-on threat"
    assert cmd["vz"] > 0, "escape should climb"
    assert body["trigger"]["gf_spikes"] >= 1


def test_escape_is_in_time(client: TestClient) -> None:
    """The calibrated loom gain leaves enough time to stop. This is the number that matters."""
    r = client.post(
        "/reflex/test",
        json={"distance_m": 6.0, "speed_mps": 5.0, "object_size_m": 0.5, "bearing_deg": 0.0},
    )
    body = r.json()
    assert body["trigger"], "reflex never fired"
    assert body["trigger"]["ttc_at_trigger_s"] >= body["required_ttc_s"], (
        f"fired {body['trigger']['ttc_at_trigger_s']} s before contact, "
        f"needs {body['required_ttc_s']} s: {body['verdict']}"
    )


def test_escape_direction_follows_the_bearing(client: TestClient) -> None:
    """A threat on the right pushes the drone left, and the mirror case is symmetric."""
    out = {}
    for bearing in (-60.0, 60.0):
        r = client.post("/reflex/test", json={"distance_m": 5.0, "speed_mps": 5.0,
                                              "object_size_m": 0.5, "bearing_deg": bearing})
        trig = r.json()["trigger"]
        assert trig, f"no trigger at bearing {bearing}"
        out[bearing] = trig["command"]["vy"]
    assert out[60.0] > 0, "threat on the right should push left (+vy)"
    assert out[-60.0] < 0, "threat on the left should push right (-vy)"


def test_reflex_events_logged(client: TestClient) -> None:
    """Escapes are logged with the geometry that caused them."""
    r = client.get("/reflex/events", params={"limit": 50})
    assert r.status_code == 200, r.text
    events = r.json()["events"]
    assert events, "no reflex events recorded"
    escapes = [e for e in events if e["kind"] == "ESCAPE"]
    assert escapes
    assert escapes[-1]["gf_spikes"] >= 1
    assert escapes[-1]["command"] is not None


def test_motor_config_round_trip(client: TestClient) -> None:
    """Settings change, and a misspelled field is reported rather than silently dropped."""
    before = client.get("/reflex/config").json()
    assert before["reflex"]["direction_source"] == "vision"
    assert before["safety"]["max_speed"] > 0
    r = client.patch(
        "/reflex/config",
        json={"reflex": {"speed_mps": 4.0}, "encoder_gains": {"loom_speed": 12.0},
              "pursuit": {"nope": 1}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reflex"]["speed_mps"] == 4.0
    assert body["unknown_fields"] == ["pursuit.nope"]
    client.patch("/reflex/config", json={"reflex": {"speed_mps": before["reflex"]["speed_mps"]}})


def test_output_gate_and_stop(client: TestClient) -> None:
    """Arming is explicit, and stop always disarms."""
    assert client.post("/drone/output", json={"enabled": True}).json()["output_enabled"] is True
    assert client.get("/drone/state").json()["output_enabled"] is True
    body = client.post("/drone/stop", json={}).json()
    assert body["output_enabled"] is False
    assert body["command"]["vx"] == 0.0


def test_pursuit_gate_closed_means_hover(client: TestClient) -> None:
    """With the courtship gate shut the drone must not translate."""
    r = client.post("/drone/pursuit", json={"arousal": 0.0})
    assert r.status_code == 200, r.text
    assert r.json()["decoder_wired"] is False
    percept = {"has_target": True, "bearing_deg": 20.0, "size_deg": 4.0, "altitude_m": 3.0}
    step = client.post("/brain/step", json={"percept": percept, "ticks": 1}).json()
    cmd = step["steps"][0]["command"]
    assert cmd["vx"] == 0.0 and cmd["vy"] == 0.0
    assert cmd["state"] in ("HOLD", "COOLDOWN")


def test_pursuit_uncalibrated_says_so(client: TestClient) -> None:
    """Opening the gate with no steer/forward types must not silently do nothing."""
    r = client.post("/drone/pursuit", json={"arousal": 1.0})
    assert "update_circuit" in r.json()["note"]
    percept = {"has_target": True, "bearing_deg": 20.0, "size_deg": 4.0, "altitude_m": 3.0}
    step = client.post("/brain/step", json={"percept": percept, "ticks": 1}).json()
    cmd = step["steps"][0]["command"]
    assert cmd["vx"] == 0.0, "uncalibrated pursuit must hover"
    if cmd["state"] == "PURSUE":
        assert "not calibrated" in cmd["reason"]
    client.post("/drone/pursuit", json={"arousal": 0.0})


def test_safety_clamp_holds(client: TestClient) -> None:
    """No configuration can talk the controller past the velocity envelope."""
    cfg = client.get("/reflex/config").json()
    client.patch("/reflex/config", json={"reflex": {"speed_mps": 99.0}})
    try:
        r = client.post("/reflex/test", json={"distance_m": 5.0, "speed_mps": 5.0,
                                              "object_size_m": 0.5, "bearing_deg": 0.0})
        trig = r.json()["trigger"]
        assert trig, "no trigger"
        cmd = trig["command"]
        assert (cmd["vx"] ** 2 + cmd["vy"] ** 2) ** 0.5 <= cfg["safety"]["max_speed"] + 1e-6
        assert abs(cmd["vz"]) <= cfg["safety"]["max_climb"] + 1e-6
    finally:
        client.patch("/reflex/config", json={"reflex": {"speed_mps": cfg["reflex"]["speed_mps"]}})


@pytest.mark.skipif(not IS_FLYCNS, reason="needs the flycns backend and a real graph")
def test_find_neurons(client: TestClient) -> None:
    """Regex search finds the split subtypes that exact-name matching would miss."""
    r = client.get("/brain/neurons", params={"pattern": "^pC1", "limit": 50})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_matching_types"] > 1
    assert all(t.startswith("pC1") for t in body["types"])


@pytest.mark.skipif(not IS_FLYCNS, reason="needs the flycns backend and a real graph")
def test_stimulate_drives_the_escape_pathway(client: TestClient) -> None:
    """Driving the loom inputs directly makes the giant fibre and its motor targets fire."""
    r = client.post("/brain/stimulate", json={"types": ["LC4", "LPLC2"], "hz": 20.0, "ms": 200.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("spikes"), body
    assert body["spikes"].get("DNp01", 0) > 0, f"GF silent: {body}"
    assert set(body["stimulated"]) == {"LC4", "LPLC2"}


def test_stimulate_on_mock_explains_itself(client: TestClient) -> None:
    """On the mock brain stimulate has nothing to do, and says so instead of erroring."""
    if IS_FLYCNS:
        pytest.skip("mock-only")
    body = client.post("/brain/stimulate", json={"types": ["LC4"]}).json()
    assert "mock" in body.get("note", "")


def test_mcp_tools_take_flat_arguments_and_no_arguments() -> None:
    """Tools must expose named fields, and must work when called with no arguments at all.

    Both halves of this have broken before. FastMCP flattens a request body into named tool
    parameters only while the body is a plain model, so wrapping one in ``Model | None``
    collapses the whole schema into a single opaque ``req`` object. But a body whose every
    field has a default must still be optional, because FastMCP sends NO body at all when the
    argument dict is empty, and a required body then fails with HTTP 422. The fix that
    satisfies both is a plain model with a default instance. Testing over HTTP alone misses
    this entirely, because a TestClient always sends ``json={}``.
    """

    async def go() -> None:
        async with Client(mcp) as c:
            tools = {t.name: t for t in await c.list_tools()}
            flat = {
                "load_brain": "backend",
                "step_brain": "percept",
                "test_reflex": "distance_m",
                "update_motor_config": "reflex",
                "set_pursuit": "arousal",
                "stimulate_neurons": "types",
                "enable_output": "enabled",
                "find_neurons": "pattern",
            }
            for name, field in flat.items():
                schema = tools[name].input_schema or {}
                props = set(schema.get("properties", {}))
                assert field in props, f"{name} lost its flat parameters: {props}"
                assert "req" not in props, f"{name} collapsed into an opaque req object"

            # every tool whose arguments are all optional must accept an empty call
            for name in ("brain_status", "load_brain", "test_reflex", "get_circuit",
                         "get_motor_config", "update_motor_config", "set_pursuit",
                         "drone_state", "reflex_events", "stop_drone", "step_brain"):
                await c.call_tool(name, {})

    asyncio.run(go())


def test_mcp_arguments_actually_take_effect() -> None:
    """A value passed to a tool must reach the handler, not be silently dropped."""

    async def go() -> None:
        async with Client(mcp) as c:
            r = (await c.call_tool("test_reflex", {"distance_m": 8.0, "speed_mps": 4.0})).data
            assert r["stimulus"]["distance_m"] == 8.0
            assert r["stimulus"]["speed_mps"] == 4.0
            assert r["trace"][0]["distance_m"] == 8.0
            await c.call_tool("stop_drone", {})

    asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
