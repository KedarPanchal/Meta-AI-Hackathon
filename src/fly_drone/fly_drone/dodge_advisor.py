"""
Ask Muse Spark which way to dodge, ahead of time, so the answer is there when the spike comes.

The escape reflex fires in a tenth of a second; a vision model answers in seconds. So the two
are decoupled: whenever the tracker has a *moving* object (or any camera blob) the node hands
this advisor the forward-camera frame and the object's tracked motion, one request in flight
at a time and one per object every few seconds, on a thread. Muse predicts where the object
goes next and picks a dodge ('left', 'right', 'up', 'down'); the answer is cached per track
and the escape uses it if it has arrived within ``advice_max_age_s`` of firing. If it has not,
the geometric predictor in :mod:`fly_brain.dodge` decides, as it always can.

Nothing here is on the control path: a slow or failed answer costs nothing but the advice.
Needs ``MUSE_API_KEY`` or ``MODEL_API_KEY``; without a key the node does not build it.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time

from fly_brain.schemas import DodgeAdvice

SYSTEM_PROMPT = (
    'You are the evasive-manoeuvre advisor of a small quadrotor flying through a forest. You '
    'get one frame from its forward camera (60 degree field of view, the horizon near the '
    'middle) and the motion of one object its sensors are tracking: bearing (+ is right of the '
    'nose), elevation (+ is up), range, speed and the direction it has been moving. Predict '
    'where the object will be in the next two seconds and choose the dodge that best gets the '
    'drone out of its path: "left", "right", "up" or "down". Prefer sideways: climbing or '
    'diving keeps the drone under or over the object and in its downwash, and vertical moves '
    'near trees or the ground are risky. Dodge AWAY from the side the object will pass on. '
    'Answer with ONE JSON object and nothing else: {"expected_path": "one short phrase", '
    '"dodge": "left" | "right" | "up" | "down", "confidence": 0 to 1, "reason": "one short sentence"}'
)

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def describe_threat(summary: dict) -> str:
    """The tracked motion, as the user turn."""
    v = summary.get('velocity_body')
    motion = 'motion unknown'
    if v:
        motion = (f'moving {v[0]:+.1f} m/s along my nose, {v[1]:+.1f} m/s to my left, '
                  f'{v[2]:+.1f} m/s up')
    return (f"Object: {summary.get('label', '')}; bearing {summary.get('bearing_deg', 0):+.0f} deg, "
            f"elevation {summary.get('elevation_deg', 0):+.0f} deg, range "
            f"{summary.get('distance_m') or 'unknown'} m, angular size {summary.get('size_deg', 0):.1f} deg, "
            f"{motion}. I am flying at {summary.get('speed_mps', 0):.1f} m/s. Which way do I dodge?")


def parse_advice(text: str, track_key: str, now: float) -> DodgeAdvice | None:
    body = (text or '').strip()
    if body.startswith('```'):
        body = re.sub(r'^```[a-zA-Z]*\s*', '', body)
        body = re.sub(r'\s*```$', '', body)
    data = None
    m = _JSON_RE.search(body)
    for cand in ([body] + ([m.group(0)] if m else [])):
        try:
            data = json.loads(cand)
            break
        except Exception:
            continue
    if not isinstance(data, dict):
        return None
    direction = str(data.get('dodge', '')).strip().lower()
    if direction not in ('left', 'right', 'up', 'down'):
        return None
    try:
        conf = float(data.get('confidence', 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return DodgeAdvice(
        axis='lateral' if direction in ('left', 'right') else 'vertical', direction=direction,
        expected=str(data.get('expected_path', ''))[:120], confidence=max(0.0, min(1.0, conf)),
        source='muse', reason=str(data.get('reason', ''))[:200], track_key=track_key, t=now,
    )


def encode_jpeg(image_bgr, max_width: int = 640, quality: int = 80) -> bytes:
    import cv2
    import numpy as np

    img = np.ascontiguousarray(image_bgr)
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, max(1, int(h * max_width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return bytes(buf)


class MuseDodgeAdvisor:
    """One request in flight at a time; answers cached per track key."""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout_s: float = 45.0, min_interval_s: float = 4.0,
                 transport=None):
        import httpx

        self.api_key = api_key or os.environ.get('MUSE_API_KEY') or os.environ.get('MODEL_API_KEY') or ''
        if not self.api_key:
            raise RuntimeError('the dodge advisor needs MUSE_API_KEY (or MODEL_API_KEY)')
        self.model = model or os.environ.get('MUSE_MODEL', 'muse-spark-1.3')
        base = (base_url or os.environ.get('MUSE_BASE_URL', 'https://api.meta.ai/v1')).rstrip('/')
        self.url = base + '/chat/completions'
        self.min_interval_s = min_interval_s
        # Optional request fields, dropped for good on a 400 (as the mission detector does).
        # reasoning_effort matters: with a frame and the motion text, muse-spark-1.3 reasoned
        # past 1200 tokens at the default effort and answered nothing (measured 2026-11-03).
        self._extras: dict = {'reasoning_effort': os.environ.get('MUSE_REASONING_EFFORT', 'low')}
        self.last_usage: dict | None = None
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s), transport=transport,
            headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
        )
        self._lock = threading.Lock()
        self._busy = False
        self._asked: dict[str, float] = {}
        self._advice: dict[str, DodgeAdvice] = {}
        self.calls = 0
        self.errors = 0
        self.last_error: str | None = None

    # --- the slow path, on its own thread ---------------------------------------------

    def consider(self, frame_bgr, summary: dict, track_key: str, now: float) -> bool:
        """Ask about this object if nothing is in flight and it was not asked recently."""
        with self._lock:
            if self._busy or now - self._asked.get(track_key, -1e9) < self.min_interval_s:
                return False
            self._busy = True
            self._asked[track_key] = now
        threading.Thread(target=self._ask, args=(frame_bgr, summary, track_key),
                         name='muse-dodge', daemon=True).start()
        return True

    def _ask(self, frame_bgr, summary: dict, track_key: str) -> None:
        try:
            advice = self.ask(frame_bgr, summary, track_key)
            if advice is not None:
                with self._lock:
                    self._advice[track_key] = advice
        finally:
            with self._lock:
                self._busy = False

    def ask(self, frame_bgr, summary: dict, track_key: str) -> DodgeAdvice | None:
        """One blocking call. Returns None on any failure; the failure is in ``last_error``."""
        self.calls += 1
        try:
            content = [{'type': 'text', 'text': describe_threat(summary)}]
            if frame_bgr is not None:
                b64 = base64.b64encode(encode_jpeg(frame_bgr)).decode('ascii')
                content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}})
            body = {
                # a reasoning model: its thinking counts against max_tokens, so leave room
                'model': self.model, 'temperature': 0, 'max_tokens': 2500,
                'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                             {'role': 'user', 'content': content}],
            }
            r = self._client.post(self.url, json={**body, **self._extras})
            if r.status_code == 400 and self._extras:
                self._extras = {}
                r = self._client.post(self.url, json=body)
            if r.status_code != 200:
                raise RuntimeError(f'HTTP {r.status_code}: {r.text[:200]}')
            data = r.json()
            self.last_usage = data.get('usage')
            choice = data['choices'][0]
            text = choice['message']['content']
            if isinstance(text, list):
                text = ''.join(p.get('text', '') for p in text if isinstance(p, dict))
            if not text:
                raise RuntimeError(f"empty answer (finish_reason {choice.get('finish_reason')}): out of tokens while reasoning")
            advice = parse_advice(text, track_key, time.time())
            if advice is None:
                raise RuntimeError(f'unparseable answer: {str(text)[:120]}')
            return advice
        except Exception as exc:
            self.errors += 1
            self.last_error = f'{type(exc).__name__}: {exc}'
            return None

    # --- the fast path ---------------------------------------------------------------------

    def latest(self, track_key: str | None = None) -> DodgeAdvice | None:
        with self._lock:
            if track_key is not None:
                return self._advice.get(track_key)
            return max(self._advice.values(), key=lambda a: a.t) if self._advice else None
