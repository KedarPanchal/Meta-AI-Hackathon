"""Muse's general navigation contract, against a fake Meta Model API."""
from __future__ import annotations

import json

import httpx
import numpy as np

from mission.config import MissionSettings
from mission.muse_navigator import MuseNavigator, parse_nav_decision


def cfg() -> MissionSettings:
    c = MissionSettings()
    c.muse_api_key = 'test-key'
    c.muse_model = 'muse-spark-test'
    return c


def frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def test_muse_can_choose_a_candidate_goal() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen['body'] = json.loads(request.content)
        answer = ('{"action":"goto","target_id":"0:1:1.0:2.0","target_pixel":null,'
                  '"confidence":0.8,"reason":"this lane is closest"}')
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                                       'message': {'content': answer}}]})

    nav = MuseNavigator(cfg(), transport=httpx.MockTransport(handler))
    d = nav.choose(frame(), 15.0, [{'id': '0:1:1.0:2.0', 'x': 1.0, 'y': 2.0, 'z': 15.0}])
    assert d.error is None and d.action == 'goto' and d.target_id == '0:1:1.0:2.0'
    body = seen['body']
    assert body['max_tokens'] >= 1000
    assert body['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,')


def test_muse_can_investigate_a_pixel_and_geometry_is_general() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        answer = ('{"action":"investigate","target_id":null,"target_pixel":[0.5,0.2],'
                  '"confidence":0.9,"reason":"a person under the trees"}')
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                                       'message': {'content': answer}}]})

    nav = MuseNavigator(cfg(), transport=httpx.MockTransport(handler))
    d = nav.choose(frame(), 15.0, [])
    assert d.error is None and d.action == 'investigate'
    assert d.target_pixel == (0.5, 0.2)


class RawSink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def publish(self, role, model, raw, **kw) -> None:
        self.records.append({'role': role, 'model': model, 'raw': raw, **kw})


def test_the_complete_answer_goes_to_the_raw_log_topic() -> None:
    answer = ('{"action":"hold","target_id":null,"target_pixel":null,'
              '"confidence":0.2,"reason":"nothing useful in view"}')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                                       'message': {'content': answer}}]})

    sink = RawSink()
    nav = MuseNavigator(cfg(), transport=httpx.MockTransport(handler), raw_sink=sink)
    d = nav.choose(frame(), 15.0, [])
    assert d.error is None and d.action == 'hold'
    assert len(sink.records) == 1
    rec = sink.records[0]
    assert rec['role'] == 'navigator' and rec['model'] == 'muse-spark-test'
    assert rec['ok'] is True and rec['raw'] == answer


def test_unknown_candidate_is_an_error_not_a_guess() -> None:
    d = parse_nav_decision('{"action":"goto","target_id":"made-up","target_pixel":null,'
                           '"confidence":0.5,"reason":"x"}', {'real'})
    assert d.error is not None and 'unknown candidate' in d.error
    assert d.action == 'hold'          # fail closed: never fly to an invented id


def test_a_bad_answer_does_not_become_a_miss() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                                       'message': {'content': None}}]})

    nav = MuseNavigator(cfg(), transport=httpx.MockTransport(handler))
    d = nav.choose(frame(), 15.0, [])
    assert d.error is not None and nav.errors == 1
