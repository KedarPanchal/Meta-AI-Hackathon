"""The Muse dodge advisor: the request it builds, the answer it parses, its caching."""
from __future__ import annotations

import json

from fly_drone.dodge_advisor import MuseDodgeAdvisor, describe_threat, parse_advice

import httpx

import numpy as np


def test_parse_advice_forms() -> None:
    a = parse_advice('{"expected_path": "veers left", "dodge": "right", "confidence": 0.8, "reason": "x"}', 'lidar:3', 10.0)
    assert a is not None and a.direction == 'right' and a.axis == 'lateral' and a.source == 'muse'
    assert a.track_key == 'lidar:3' and a.t == 10.0 and a.confidence == 0.8
    a = parse_advice('```json\n{"dodge": "Up"}\n```', 'c:1', 0.0)
    assert a is not None and a.direction == 'up' and a.axis == 'vertical' and a.confidence == 0.5
    assert parse_advice('I would go sideways', 'c:1', 0.0) is None
    assert parse_advice('{"dodge": "backwards"}', 'c:1', 0.0) is None


def test_describe_threat_mentions_the_motion() -> None:
    s = describe_threat({'label': 'sphere', 'bearing_deg': 12.0, 'elevation_deg': -3.0, 'distance_m': 7.5,
                         'size_deg': 4.1, 'velocity_body': (-4.8, 0.6, 0.0), 'speed_mps': 2.5})
    assert 'bearing +12 deg' in s and '-4.8 m/s along my nose' in s and '+0.6 m/s to my left' in s


def test_advisor_round_trip_with_a_fake_server() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen['body'] = body
        answer = '{"expected_path": "straight at me then up", "dodge": "left", "confidence": 0.7, "reason": "it climbs"}'
        return httpx.Response(200, json={'choices': [{'message': {'content': answer}}]})

    adv = MuseDodgeAdvisor(api_key='k', transport=httpx.MockTransport(handler), min_interval_s=100.0)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    a = adv.ask(frame, {'label': 'sphere', 'bearing_deg': 0, 'elevation_deg': 0, 'distance_m': 8,
                        'size_deg': 3.5, 'velocity_body': (-5, 0, 0), 'speed_mps': 2.5}, 'lidar:9')
    assert a is not None and a.direction == 'left' and a.track_key == 'lidar:9'
    msgs = seen['body']['messages']
    assert msgs[0]['role'] == 'system' and 'Prefer sideways' in msgs[0]['content']
    parts = msgs[1]['content']
    assert parts[0]['type'] == 'text' and parts[1]['image_url']['url'].startswith('data:image/jpeg;base64,')
    # the threaded path: one in flight, then cached per key, not re-asked within the interval
    assert adv.consider(frame, {'label': 's'}, 'lidar:10', now=0.0) is True
    import time
    for _ in range(50):
        if adv.latest('lidar:10') is not None:
            break
        time.sleep(0.05)
    assert adv.latest('lidar:10') is not None and adv.latest('lidar:10').direction == 'left'
    assert adv.consider(frame, {'label': 's'}, 'lidar:10', now=1.0) is False


def test_the_complete_answer_goes_to_the_raw_log_callback() -> None:
    answer = '{"expected_path": "past my left", "dodge": "right", "confidence": 0.6, "reason": "x"}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'choices': [{'message': {'content': answer}}]})

    seen: list[dict] = []
    adv = MuseDodgeAdvisor(api_key='k', transport=httpx.MockTransport(handler), on_raw=seen.append)
    assert adv.ask(None, {'label': 's'}, 'lidar:4') is not None
    assert len(seen) == 1
    rec = seen[0]
    assert rec['role'] == 'dodge_advisor' and rec['model'] == adv.model
    assert rec['ok'] is True and rec['raw'] == answer
    assert rec['context'] == {'track_key': 'lidar:4'}


def test_advisor_failure_is_quiet() -> None:
    adv = MuseDodgeAdvisor(api_key='k', transport=httpx.MockTransport(lambda r: httpx.Response(500, text='boom')))
    assert adv.ask(None, {'label': 's'}, 'c:1') is None
    assert adv.errors == 1 and 'HTTP 500' in (adv.last_error or '')


def test_empty_answer_is_an_error_not_silence() -> None:
    def handler(request):
        return httpx.Response(200, json={'choices': [{'finish_reason': 'length', 'message': {'content': None, 'role': 'assistant'}}],
                                         'usage': {'completion_tokens_details': {'reasoning_tokens': 200}}})
    adv = MuseDodgeAdvisor(api_key='k', transport=httpx.MockTransport(handler))
    assert adv.ask(None, {'label': 's'}, 'c:1') is None
    assert 'empty answer' in (adv.last_error or '')
