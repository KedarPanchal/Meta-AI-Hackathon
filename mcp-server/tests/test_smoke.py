"""The MCP surface over both faces of the server: HTTP and the in-process FastMCP client.

Runs against the in-process sim link and the colour detector (see conftest.py), so it needs
neither Gazebo nor a Muse key::

    python tests/test_smoke.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('BRAIN_BACKEND', 'mock')
os.environ.setdefault('MISSION_LINK', 'sim')
os.environ.setdefault('PERSON_DETECTOR', 'color')
os.environ.setdefault('SIM_SPEEDUP', '40')
os.environ.setdefault('MISSION_POLL_S', '0.01')

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from fastmcp import Client  # noqa: E402

from app.main import server  # noqa: E402
from mcp_server import mcp  # noqa: E402
from mission.runner import reset_service  # noqa: E402

EXPECTED_TOOLS = {'search_for_person', 'scout_area', 'mission_status', 'abort_mission', 'look_for_person'}


def payload(result) -> dict:
    """The tool's JSON as a plain dict. FastMCP's ``.data`` rebuilds typed outputs into objects."""
    sc = getattr(result, 'structured_content', None)
    if isinstance(sc, dict):
        return sc
    if isinstance(result.data, dict):
        return result.data
    return json.loads(result.content[0].text)


@pytest.fixture(scope='module')
def client() -> TestClient:
    reset_service()
    return TestClient(server)


def test_health(client: TestClient) -> None:
    r = client.get('/health')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['ok'] is True
    assert body['link'] == 'sim'
    assert body['topics']['goal'] == '/fly_brain/goal'


def test_mcp_surface() -> None:
    """Exactly the five mission tools, the one prompt, and /health excluded."""

    async def go() -> None:
        async with Client(mcp) as c:
            tools = {t.name for t in await c.list_tools()}
            assert tools == EXPECTED_TOOLS, f'unexpected tool set: {tools ^ EXPECTED_TOOLS}'
            assert 'health' not in tools
            prompts = {p.name for p in await c.list_prompts()}
            assert prompts == {'find_person'}
            assert await c.list_resources() == []

    asyncio.run(go())


def test_tools_take_flat_arguments_and_no_arguments() -> None:
    """Named fields, not an opaque ``req`` object, and every all-optional tool accepts {}.

    FastMCP flattens a request body into named parameters only while the body is a plain
    model with a default instance; ``Model | None`` collapses the schema and a required body
    fails with 422 when the client sends no arguments at all.
    """

    async def go() -> None:
        async with Client(mcp) as c:
            tools = {t.name: t for t in await c.list_tools()}
            for name in ('search_for_person', 'scout_area'):
                props = set((tools[name].input_schema or {}).get('properties', {}))
                assert {'area', 'altitude_m', 'dry_run', 'keepouts'} <= props, f'{name}: {props}'
                assert 'req' not in props
            assert 'return_home' in (tools['scout_area'].input_schema or {}).get('properties', {})
            for name in ('mission_status', 'abort_mission', 'look_for_person'):
                await c.call_tool(name, {})

    asyncio.run(go())


def test_dry_run_plans_without_flying(client: TestClient) -> None:
    r = client.post('/mission/search', json={'dry_run': True, 'area': {'x_min': -30, 'x_max': 30, 'y_min': -30, 'y_max': 30}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['dry_run'] is True
    assert body['plan']['n_lanes'] >= 2
    assert body['plan']['coverage_estimate'] >= 0.97
    assert body['waypoints'][0]['z'] == 15.0
    assert client.get('/mission/status').json()['state'] == 'idle'


def test_bad_area_is_a_validation_error(client: TestClient) -> None:
    r = client.post('/mission/search', json={'dry_run': True, 'area': {'x_min': 10, 'x_max': 0, 'y_min': 0, 'y_max': 1}})
    assert r.status_code == 422


def test_mcp_arguments_actually_take_effect() -> None:
    async def go() -> None:
        async with Client(mcp) as c:
            r = payload(await c.call_tool('scout_area', {'dry_run': True, 'altitude_m': 9.0, 'lane_overlap': 0.5,
                                                         'area': {'x_min': -20, 'x_max': 20, 'y_min': -20, 'y_max': 20}}))
            assert r['dry_run'] is True
            assert r['plan']['altitude_m'] == 9.0
            assert r['plan']['lane_spacing_m'] == pytest.approx(r['plan']['footprint_m'][0] * 0.5, abs=0.1)

    asyncio.run(go())


def test_fly_a_small_search_over_mcp() -> None:
    """Start, watch, and stop a mission entirely through tool calls."""

    async def go() -> None:
        async with Client(mcp) as c:
            r = payload(await c.call_tool('search_for_person', {'area': {'x_min': -58, 'x_max': 58, 'y_min': -20, 'y_max': 58},
                                                                'keepouts': []}))
            assert 'error' not in r, r
            assert r['state'] in ('planned', 'running')
            assert r['link'] == 'sim' and r['detector'].startswith('color')
            t0 = time.time()
            st = None
            while time.time() - t0 < 10.0:
                st = payload(await c.call_tool('mission_status', {}))
                if st['state'] == 'running' and st['brain'] and st['brain']['active']:
                    break
                await asyncio.sleep(0.1)
            assert st is not None and st['state'] == 'running', st
            assert st['brain']['output_enabled'] is True
            assert st['n_waypoints'] > 0 and st['plan']['n_lanes'] >= 2
            look = payload(await c.call_tool('look_for_person', {}))
            assert 'detection' in look
            stop = payload(await c.call_tool('abort_mission', {}))
            assert stop['state'] == 'aborted'
            st = payload(await c.call_tool('mission_status', {}))
            assert st['state'] == 'aborted'

    asyncio.run(go())


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v', '--no-header', '-p', 'no:cacheprovider']))
