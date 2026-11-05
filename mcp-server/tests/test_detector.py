"""The Muse detector against a fake Meta Model API: the request, the answer, the failures."""
from __future__ import annotations

import json

import httpx
import numpy as np

from mission.config import MissionSettings
from mission.detector import MUSE_MAX_TOKENS, MuseDetector


def cfg() -> MissionSettings:
    c = MissionSettings()
    c.muse_api_key = 'test-key'
    return c


def frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def test_json_answer_becomes_a_detection() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen['body'] = json.loads(request.content)
        answer = '{"person_visible": true, "confidence": 0.9, "bbox": [0.4, 0.5, 0.5, 0.55], "description": "a person lying down"}'
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': answer}}]})

    det = MuseDetector(cfg(), transport=httpx.MockTransport(handler))
    d = det.detect(frame(), 15.0)
    assert d.error is None and d.person_visible and d.confidence == 0.9
    assert d.bbox is not None and d.centre() is not None
    body = seen['body']
    assert body['max_tokens'] == MUSE_MAX_TOKENS >= 1000, 'a reasoning model thinks against max_tokens'
    assert body['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,')


def test_empty_answer_is_an_error_not_a_miss() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            'choices': [{'finish_reason': 'length', 'message': {'content': None, 'role': 'assistant'}}],
            'usage': {'completion_tokens_details': {'reasoning_tokens': 300}},
        })

    det = MuseDetector(cfg(), transport=httpx.MockTransport(handler))
    d = det.detect(frame(), 15.0)
    assert d.error is not None and 'empty answer' in d.error and '300 reasoning tokens' in d.error
    assert not d.person_visible and det.errors == 1


class RawSink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def publish(self, role, model, raw, **kw) -> None:
        self.records.append({'role': role, 'model': model, 'raw': raw, **kw})


def test_the_complete_answer_goes_to_the_raw_log_topic() -> None:
    answer = ('{"person_visible": false, "confidence": 0.1, "bbox": null, '
              '"description": "grass"}' + ' ' * 600)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                                       'message': {'content': answer}}]})

    sink = RawSink()
    det = MuseDetector(cfg(), transport=httpx.MockTransport(handler), raw_sink=sink)
    d = det.detect(frame(), 15.0)
    assert d.error is None
    assert len(sink.records) == 1
    rec = sink.records[0]
    assert rec['role'] == 'detector' and rec['model'] == det.model
    assert rec['ok'] is True and rec['error'] is None and rec['raw'] == answer
    assert rec['height_m'] == 15.0


def test_a_400_drops_the_optional_fields_and_retries() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(sorted(body))
        if 'response_format' in body:
            return httpx.Response(400, text='unknown field response_format')
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '{"person_visible": false, "confidence": 0.1, "bbox": null, "description": "grass"}'}}]})

    det = MuseDetector(cfg(), transport=httpx.MockTransport(handler))
    d = det.detect(frame(), 15.0)
    assert d.error is None and not d.person_visible and len(calls) == 2
    assert 'response_format' in calls[0] and 'response_format' not in calls[1]
