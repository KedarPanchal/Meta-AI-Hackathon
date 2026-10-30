"""
Is there a person in this frame? Three answers to the same question.

* :class:`MuseDetector` asks Meta's **Muse Spark** through the Meta Model API, which is
  OpenAI-compatible: ``POST {base}/chat/completions`` with the frame as a base64 JPEG
  ``image_url`` content part. Default model ``muse-spark-1.3``; key from ``MUSE_API_KEY`` or
  Meta's own ``MODEL_API_KEY``. The reply is constrained to one JSON object and parsed
  defensively, because a vision model that writes "Yes, there is a person" in prose instead of
  JSON must still count as a detection and not as an error.
* :class:`ColorDetector` thresholds a colour in HSV. It is what the in-process sim is tested
  with, and it is honest about being a stand-in: ``name`` says so in every status.
* :class:`NullDetector` always says no, for scouting without a key.

Every detector returns a :class:`Detection`; failures come back **as a Detection with
``error`` set**, never as an exception, so one dropped API call does not stop a flight.
"""
from __future__ import annotations

import base64
import json
import re
import time
from typing import Protocol, runtime_checkable

from .config import MissionSettings
from .schemas import Detection

PERSON_SYSTEM_PROMPT = (
    'You are the vision module of a search-and-rescue drone. Each image is one frame from a '
    'camera pointing straight down at a forest from about {height:.0f} m above the ground; '
    'the top of the image is the direction the drone is flying. Your only job is to decide '
    'whether a HUMAN BEING is visible: standing, sitting, lying down, partly hidden under '
    'trees, any clothing. Rocks, fallen logs, tents, picnic tables, shadows and animals are '
    'not people. Be careful but not paranoid: a missing person is often lying on the ground. '
    'Answer with ONE JSON object and nothing else, exactly in this form: '
    '{{"person_visible": true or false, "confidence": number from 0 to 1, '
    '"bbox": [x0, y0, x1, y1] as fractions of image width/height with origin top-left, or null, '
    '"description": "one short sentence saying what you see"}}'
)

PERSON_USER_PROMPT = 'Is a person visible in this frame? Reply with the JSON object only.'

JSON_SCHEMA = {
    'name': 'person_detection',
    'strict': True,
    'schema': {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'person_visible': {'type': 'boolean'},
            'confidence': {'type': 'number'},
            'bbox': {
                'anyOf': [
                    {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4},
                    {'type': 'null'},
                ]
            },
            'description': {'type': 'string'},
        },
        'required': ['person_visible', 'confidence', 'bbox', 'description'],
    },
}


@runtime_checkable
class PersonDetector(Protocol):
    name: str

    def detect(self, image_bgr, height_m: float) -> Detection: ...


# --- image encoding -------------------------------------------------------------


def encode_jpeg(image_bgr, max_width: int = 768, quality: int = 85) -> bytes:
    """BGR array to JPEG bytes, downscaled to ``max_width`` so a frame is a few tens of kB."""
    try:
        import cv2
        import numpy as np

        img = np.ascontiguousarray(image_bgr)
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            img = cv2.resize(img, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError('cv2.imencode failed')
        return bytes(buf)
    except ImportError:                                # pragma: no cover - cv2 is in the env
        import io

        from PIL import Image

        img = Image.fromarray(image_bgr[:, :, ::-1])
        if img.width > max_width:
            img = img.resize((max_width, int(img.height * max_width / img.width)))
        out = io.BytesIO()
        img.save(out, format='JPEG', quality=quality)
        return out.getvalue()


# --- parsing the model's answer ---------------------------------------------------

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)
_YES_RE = re.compile(r'\b(yes|there is a person|person (is )?visible|i (can )?see a (person|human|man|woman))\b', re.I)


def parse_detection(text: str, detector: str) -> Detection:
    """
    Pull a Detection out of whatever the model wrote.

    Strict JSON first (fences stripped), then the first ``{...}`` block, then a prose fallback
    that counts an unambiguous "yes" as a low-confidence detection. Confidence is clamped;
    an invalid bbox is dropped rather than failing the whole frame.
    """
    raw = (text or '').strip()
    body = raw
    if body.startswith('```'):
        body = re.sub(r'^```[a-zA-Z]*\s*', '', body)
        body = re.sub(r'\s*```$', '', body)
    data = None
    block = _JSON_RE.search(body)
    for candidate in ([body] + ([block.group(0)] if block else [])):
        try:
            data = json.loads(candidate)
            break
        except Exception:
            continue
    if isinstance(data, dict):
        visible = bool(data.get('person_visible', data.get('person', False)))
        try:
            conf = float(data.get('confidence', 0.5 if visible else 0.0))
        except (TypeError, ValueError):
            conf = 0.5 if visible else 0.0
        conf = max(0.0, min(1.0, conf))
        bbox = None
        b = data.get('bbox')
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                vals = [float(v) for v in b]
                if max(vals) > 1.5:                     # pixels, not fractions: cannot place it
                    vals = []
                if vals:
                    x0, y0, x1, y1 = vals
                    bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                    bbox = tuple(max(0.0, min(1.0, v)) for v in bbox)  # type: ignore[assignment]
            except (TypeError, ValueError):
                bbox = None
        return Detection(
            person_visible=visible, confidence=conf, bbox=bbox,
            description=str(data.get('description', ''))[:300], detector=detector, raw=raw[:500],
        )
    # prose fallback
    visible = bool(_YES_RE.search(raw)) and not re.search(r'\bno (person|one|human)\b', raw, re.I)
    return Detection(
        person_visible=visible, confidence=0.4 if visible else 0.0, bbox=None,
        description=raw[:300], detector=detector, raw=raw[:500],
    )


# --- the detectors -------------------------------------------------------------------


class MuseDetector:
    """Muse Spark over the Meta Model API. One HTTP call per frame."""

    def __init__(self, cfg: MissionSettings):
        import httpx

        if not cfg.muse_api_key:
            raise RuntimeError(
                'PERSON_DETECTOR=muse needs MUSE_API_KEY (or MODEL_API_KEY) from the Meta Model '
                'API dashboard.'
            )
        self.cfg = cfg
        self.model = cfg.muse_model
        self.name = f'muse:{self.model}'
        self.url = cfg.muse_base_url.rstrip('/') + '/chat/completions'
        self._client = httpx.Client(
            timeout=httpx.Timeout(cfg.muse_timeout_s),
            headers={'Authorization': f'Bearer {cfg.muse_api_key}', 'Content-Type': 'application/json'},
        )
        # Optional request fields. If the API rejects one with a 400, it is dropped for good and
        # the call is retried once without it, so an incompatible extra never costs a mission.
        self._extras: dict = {}
        self._extras['response_format'] = {'type': 'json_schema', 'json_schema': JSON_SCHEMA}
        if cfg.muse_reasoning_effort:
            self._extras['reasoning_effort'] = cfg.muse_reasoning_effort
        self.calls = 0
        self.errors = 0
        self.last_error: str | None = None

    def _body(self, image_b64: str, height_m: float) -> dict:
        body = {
            'model': self.model,
            'temperature': 0,
            'max_tokens': 300,
            'messages': [
                {'role': 'system', 'content': PERSON_SYSTEM_PROMPT.format(height=height_m)},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': PERSON_USER_PROMPT},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}},
                    ],
                },
            ],
        }
        body.update(self._extras)
        return body

    def detect(self, image_bgr, height_m: float) -> Detection:
        import httpx

        t0 = time.perf_counter()
        self.calls += 1
        try:
            jpeg = encode_jpeg(image_bgr, max_width=self.cfg.muse_max_image_w)
        except Exception as exc:
            return self._fail(f'encode: {type(exc).__name__}: {exc}', t0)
        b64 = base64.b64encode(jpeg).decode('ascii')
        for attempt in range(3):
            try:
                r = self._client.post(self.url, json=self._body(b64, height_m))
            except httpx.HTTPError as exc:
                if attempt < 2:
                    time.sleep(1.0 + attempt)
                    continue
                return self._fail(f'{type(exc).__name__}: {exc}', t0)
            if r.status_code == 400 and self._extras:
                # most likely one of the optional fields; drop them all and go again
                self._extras = {}
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2.0 + 2.0 * attempt)
                continue
            if r.status_code != 200:
                return self._fail(f'HTTP {r.status_code}: {r.text[:300]}', t0)
            try:
                data = r.json()
                text = data['choices'][0]['message']['content']
                if isinstance(text, list):          # some servers return content parts
                    text = ''.join(p.get('text', '') for p in text if isinstance(p, dict))
            except Exception as exc:
                return self._fail(f'unexpected response shape: {exc}: {r.text[:300]}', t0)
            det = parse_detection(text or '', self.name)
            det.latency_ms = (time.perf_counter() - t0) * 1000.0
            return det
        return self._fail('gave up after retries', t0)

    def _fail(self, msg: str, t0: float) -> Detection:
        self.errors += 1
        self.last_error = msg
        return Detection(detector=self.name, error=msg, latency_ms=(time.perf_counter() - t0) * 1000.0)


HSV_RANGES = {
    'magenta': [((140, 90, 90), (170, 255, 255))],
    'red': [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    'blue': [((100, 120, 70), (130, 255, 255))],
    'yellow': [((20, 120, 120), (35, 255, 255))],
    'orange': [((10, 150, 120), (22, 255, 255))],
}


class ColorDetector:
    """
    A saturated-colour blob counts as a person. For the kinematic sim and for a Gazebo model
    you have textured yourself; not for the real world.
    """

    def __init__(self, color: str = 'magenta', min_area_px: int = 12):
        if color not in HSV_RANGES:
            raise ValueError(f'unknown DETECT_COLOR {color!r}; have {sorted(HSV_RANGES)}')
        self.color = color
        self.min_area_px = min_area_px
        self.name = f'color:{color} (stand-in, not a vision model)'
        self.calls = 0
        self.errors = 0
        self.last_error: str | None = None

    def detect(self, image_bgr, height_m: float) -> Detection:
        t0 = time.perf_counter()
        self.calls += 1
        try:
            import cv2
            import numpy as np
        except ImportError as exc:                 # pragma: no cover
            self.errors += 1
            self.last_error = str(exc)
            return Detection(detector=self.name, error=f'opencv missing: {exc}')
        if image_bgr is None or getattr(image_bgr, 'size', 0) == 0:
            return Detection(detector=self.name, error='empty frame')
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(np.ascontiguousarray(image_bgr), cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in HSV_RANGES[self.color]:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            a = cv2.contourArea(c)
            if a >= self.min_area_px and (best is None or a > best[0]):
                best = (a, cv2.boundingRect(c))
        latency = (time.perf_counter() - t0) * 1000.0
        if best is None:
            return Detection(detector=self.name, description='no blob', latency_ms=latency)
        a, (x, y, bw, bh) = best
        conf = max(0.55, min(0.99, 0.55 + a / 400.0))
        return Detection(
            person_visible=True, confidence=conf,
            bbox=(x / w, y / h, (x + bw) / w, (y + bh) / h),
            description=f'{self.color} blob of {int(a)} px', detector=self.name, latency_ms=latency,
        )


class NullDetector:
    name = 'none (detection disabled)'

    def __init__(self) -> None:
        self.calls = 0
        self.errors = 0
        self.last_error: str | None = None

    def detect(self, image_bgr, height_m: float) -> Detection:
        self.calls += 1
        return Detection(detector=self.name, description='detection disabled')


def make_detector(cfg: MissionSettings) -> PersonDetector:
    """``PERSON_DETECTOR``: muse | color | none | auto (muse if a key is set, else color)."""
    kind = (cfg.detector or 'auto').lower()
    if kind == 'auto':
        kind = 'muse' if cfg.muse_api_key else 'color'
    if kind == 'muse':
        return MuseDetector(cfg)
    if kind == 'color':
        return ColorDetector(cfg.detect_color)
    if kind == 'none':
        return NullDetector()
    raise ValueError(f'unknown PERSON_DETECTOR {kind!r}; choose muse, color, none or auto')
