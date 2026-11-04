"""
Muse Spark as the mission's navigator, not a scenario table.

The mission layer gives Muse a downward frame and a short list of real candidate goals. Muse
chooses one of three general actions:

* ``goto`` a candidate id (an ordinary coverage waypoint, chosen by the model);
* ``investigate`` a pixel in the current frame (the model's own person target); or
* ``hold`` while none of the visible options is worth flying to.

There is no ``forest`` branch, no ``if person found then ...`` here. The same contract works
for another world, another camera height and another set of candidates. The geometry maps
Muse's chosen pixel to the ground; the brain turns the resulting world goal into velocity and
keeps its escape reflex and safety clamp. Muse never publishes a velocity.

The return type is deliberately an error-carrying model: a bad answer or a dropped HTTP call
must not stop a flight, and must not be mistaken for "nobody here".
"""
from __future__ import annotations

import base64
import json
import re
import time
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from .config import MissionSettings
from .detector import encode_jpeg

MUSE_NAV_MAX_TOKENS = 1500
"""Reasoning tokens count against this; 1500 leaves room for a one-line JSON answer."""

NAV_SYSTEM_PROMPT = """You are the navigator of a search-and-rescue drone. The image is from a
camera pointing straight down. The top of the image is the drone's forward direction. The
drone is searching for a missing person. You choose ONE general action from the options below;
do not assume a particular kind of terrain or a fixed search pattern.

Candidates are real world goals the mission planner can fly to. Choose one of their exact ids
if moving to it is a sensible next step. If you instead see a person, clothing, or a human
shape anywhere in the frame, choose investigate and give its centre as a pixel. If neither is
useful, choose hold. Never invent a candidate id. Answer with ONE JSON object and nothing else:
{{"action": "goto" | "investigate" | "hold", "target_id": string or null,
 "target_pixel": [u, v] or null, "confidence": number from 0 to 1,
 "reason": "one short sentence"}}
Pixel coordinates are fractions of image width and height, origin top-left (0..1)."""

NAV_SCHEMA = {
    'name': 'navigation_decision',
    'strict': True,
    'schema': {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'action': {'type': 'string', 'enum': ['goto', 'investigate', 'hold']},
            'target_id': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
            'target_pixel': {
                'anyOf': [
                    {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2},
                    {'type': 'null'},
                ]
            },
            'confidence': {'type': 'number'},
            'reason': {'type': 'string'},
        },
        'required': ['action', 'target_id', 'target_pixel', 'confidence', 'reason'],
    },
}

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


class NavDecision(BaseModel):
    """One general navigation choice, plus a clear failure field."""

    action: Literal['goto', 'investigate', 'hold'] = 'hold'
    target_id: str | None = None
    target_pixel: tuple[float, float] | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ''
    error: str | None = Field(default=None, description='Set when Muse failed; then do not trust the action.')
    raw: str | None = None


def parse_nav_decision(text: str, candidate_ids: set[str] | None = None) -> NavDecision:
    """Parse Muse's JSON defensively; an invalid goto is an error, not a guess."""
    raw = (text or '').strip()
    body = raw
    if body.startswith('```'):
        body = re.sub(r'^```[a-zA-Z]*\s*', '', body)
        body = re.sub(r'\s*```$', '', body)
    block = _JSON_RE.search(body)
    data = None
    for candidate in ([body] + ([block.group(0)] if block else [])):
        try:
            data = json.loads(candidate)
            break
        except Exception:
            continue
    if not isinstance(data, dict):
        return NavDecision(error=f'Muse did not return JSON: {raw[:200]}', raw=raw[:500])
    try:
        conf = max(0.0, min(1.0, float(data.get('confidence', 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    action = str(data.get('action', 'hold')).lower()
    if action not in ('goto', 'investigate', 'hold'):
        return NavDecision(error=f'unknown action {action!r}', confidence=conf, raw=raw[:500])
    target_id = data.get('target_id')
    target_id = str(target_id) if target_id is not None else None
    pixel = data.get('target_pixel')
    target_pixel = None
    if isinstance(pixel, (list, tuple)) and len(pixel) == 2:
        try:
            u, v = float(pixel[0]), float(pixel[1])
            if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
                target_pixel = (u, v)
        except (TypeError, ValueError):
            target_pixel = None
    if action == 'goto':
        if not target_id or (candidate_ids is not None and target_id not in candidate_ids):
            return NavDecision(error=f'goto references unknown candidate {target_id!r}',
                               confidence=conf, raw=raw[:500])
    if action == 'investigate' and target_pixel is None:
        return NavDecision(error='investigate without a valid normalised target_pixel',
                           confidence=conf, raw=raw[:500])
    return NavDecision(
        action=action, target_id=target_id, target_pixel=target_pixel,
        confidence=conf, reason=str(data.get('reason', ''))[:300], raw=raw[:500],
    )


class MuseNavigator:
    """The Meta Model API navigator. One call per decision, never one per control tick."""

    def __init__(self, cfg: MissionSettings, transport=None):
        if not cfg.muse_api_key:
            raise RuntimeError(
                'MISSION_NAVIGATOR=muse needs MUSE_API_KEY (or MODEL_API_KEY) from the Meta Model API.'
            )
        self.cfg = cfg
        self.name = f'muse-nav:{cfg.muse_model}'
        self.url = cfg.muse_base_url.rstrip('/') + '/chat/completions'
        self._client = httpx.Client(
            timeout=httpx.Timeout(cfg.muse_timeout_s), transport=transport,
            headers={'Authorization': f'Bearer {cfg.muse_api_key}', 'Content-Type': 'application/json'},
        )
        self._extras: dict = {
            'response_format': {'type': 'json_schema', 'json_schema': NAV_SCHEMA},
        }
        if cfg.muse_reasoning_effort:
            self._extras['reasoning_effort'] = cfg.muse_reasoning_effort
        self.calls = 0
        self.errors = 0
        self.last_error: str | None = None

    def _body(self, image_b64: str, height_m: float, candidates: list[dict]) -> dict:
        prompt = (
            f'Height above the ground: {height_m:.1f} m. '
            f'Candidate goals: {json.dumps(candidates, separators=(",", ":"))}. '
            'Choose the next action. Reply with the JSON object only.'
        )
        body = {
            'model': self.cfg.muse_model,
            'temperature': 0,
            'max_tokens': MUSE_NAV_MAX_TOKENS,
            'messages': [
                {'role': 'system', 'content': NAV_SYSTEM_PROMPT},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}},
                    ],
                },
            ],
        }
        body.update(self._extras)
        return body

    def choose(self, image_bgr, height_m: float, candidates: list[dict]) -> NavDecision:
        t0 = time.perf_counter()
        self.calls += 1
        try:
            b64 = base64.b64encode(encode_jpeg(image_bgr, max_width=self.cfg.muse_max_image_w)).decode('ascii')
        except Exception as exc:
            return self._fail(f'encode: {type(exc).__name__}: {exc}', t0)
        ids = {str(c['id']) for c in candidates}
        for attempt in range(3):
            try:
                r = self._client.post(self.url, json=self._body(b64, height_m, candidates))
            except httpx.HTTPError as exc:
                if attempt < 2:
                    time.sleep(1.0 + attempt)
                    continue
                return self._fail(f'{type(exc).__name__}: {exc}', t0)
            if r.status_code == 400 and self._extras:
                self._extras = {}
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2.0 + 2.0 * attempt)
                continue
            if r.status_code != 200:
                return self._fail(f'HTTP {r.status_code}: {r.text[:300]}', t0)
            try:
                data = r.json()
                choice = data['choices'][0]
                text = choice['message']['content']
                if isinstance(text, list):
                    text = ''.join(p.get('text', '') for p in text if isinstance(p, dict))
            except Exception as exc:
                return self._fail(f'unexpected response shape: {exc}', t0)
            if not text:
                usage = data.get('usage', {}).get('completion_tokens_details', {})
                return self._fail(
                    f"empty answer (finish_reason {choice.get('finish_reason')}, "
                    f"{usage.get('reasoning_tokens', '?')} reasoning tokens): raise MUSE_NAV_MAX_TOKENS",
                    t0,
                )
            decision = parse_nav_decision(text, ids)
            if decision.error:
                return self._fail(decision.error, t0, decision)
            return decision
        return self._fail('gave up after retries', t0)

    def _fail(self, msg: str, t0: float, decision: NavDecision | None = None) -> NavDecision:
        self.errors += 1
        self.last_error = msg
        out = decision or NavDecision()
        out.error = msg
        return out

    def close(self) -> None:
        self._client.close()
