"""
``detect(image) -> [Box]``: the hook the handoff left open, filled with a real detector.

Two strategies, because a looming threat and a courtship target look nothing alike:

* **colour** -- threshold in HSV for a known target colour. This is the one to use in Gazebo,
  where you choose the object's material yourself, and it is stable at any range.
* **motion** -- difference against a decaying background. Needs no prior about the object, but
  a moving camera smears the whole frame, so it is only honest while roughly station-keeping.

Both return boxes in pixels with a stable ``track_id``, because :class:`LoomTracker`
differentiates angular size per track and would otherwise read the gap between two objects as
an enormous expansion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os

from fly_brain.schemas import Box

# A saturated red object, the default Gazebo marker colour. Hue wraps at 180 in OpenCV, so red
# needs two ranges.
HSV_RANGES = {
    'red': [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    'green': [((40, 80, 60), (85, 255, 255))],
    'blue': [((100, 120, 70), (130, 255, 255))],
    'yellow': [((20, 120, 120), (35, 255, 255))],
}


@dataclass
class Detector:
    """Stateful so motion detection has a background and tracks keep their ids."""

    mode: str = field(default_factory=lambda: os.environ.get('FLYDRONE_DETECT', 'color'))
    color: str = field(default_factory=lambda: os.environ.get('FLYDRONE_DETECT_COLOR', 'red'))
    min_area_px: int = 24
    max_boxes: int = 4
    background_alpha: float = 0.1
    _bg = None
    _next_id: int = 0
    _tracks: dict[int, tuple[float, float]] = field(default_factory=dict)

    # --- entry point ---------------------------------------------------------

    def detect(self, image) -> list[Box]:
        """``image`` is an HxWx3 BGR array. Returns the largest few blobs as boxes."""
        try:
            import cv2
            import numpy as np
        except ImportError:                     # pragma: no cover - opencv absent
            return []
        if image is None or getattr(image, 'size', 0) == 0:
            return []
        h, w = image.shape[:2]
        if self.mode == 'color':
            mask = self._mask_color(image, cv2, np)
        else:
            mask = self._mask_motion(image, cv2, np)
        if mask is None:
            return []
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area_px:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            found.append((bw * bh, x, y, bw, bh))
        found.sort(reverse=True)
        boxes = []
        for _, x, y, bw, bh in found[: self.max_boxes]:
            cx, cy = x + bw / 2.0, y + bh / 2.0
            boxes.append(
                Box(x=float(x), y=float(y), w=float(bw), h=float(bh), image_w=float(w),
                    image_h=float(h), label=self.color if self.mode == 'color' else 'motion',
                    track_id=self._assign(cx, cy, max(bw, bh)))
            )
        live = {b.track_id for b in boxes}
        self._tracks = {k: v for k, v in self._tracks.items() if k in live}
        return boxes

    # --- masks ---------------------------------------------------------------

    def _mask_color(self, image, cv2, np):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ranges = HSV_RANGES.get(self.color)
        if not ranges:
            raise ValueError(
                f'unknown FLYDRONE_DETECT_COLOR {self.color!r}; '
                f'have {sorted(HSV_RANGES)}'
            )
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        return mask

    def _mask_motion(self, image, cv2, np):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._bg is None or self._bg.shape != gray.shape:
            self._bg = gray
            return None                 # nothing to compare against on the first frame
        diff = cv2.absdiff(gray, self._bg).astype(np.uint8)
        self._bg = (1 - self.background_alpha) * self._bg + self.background_alpha * gray
        _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        return mask

    # --- tracking ------------------------------------------------------------

    def _assign(self, cx: float, cy: float, size: float) -> int:
        """Nearest-centroid association. Enough for one or two objects; not a real tracker."""
        best, best_d = -1, 1e18
        radius = max(size * 1.5, 32.0)
        for tid, (px, py) in self._tracks.items():
            d = (cx - px) ** 2 + (cy - py) ** 2
            if d < best_d:
                best, best_d = tid, d
        if best >= 0 and best_d <= radius**2:
            self._tracks[best] = (cx, cy)
            return best
        self._next_id += 1
        self._tracks[self._next_id] = (cx, cy)
        return self._next_id


def boxes_from_ground_truth(poses, hfov_deg: float, image_w: int = 640, image_h: int = 480,
                            object_size_m: float = 0.5) -> list[Box]:
    """
    Project known relative positions into pixel boxes: Gazebo ground truth, no vision.

    ``poses`` is a list of ``(x_forward, y_left, z_up, track_id)`` in the drone's body frame.
    Use this to separate "is the brain steering correctly" from "is the detector working".
    """
    import math

    half = math.tan(math.radians(hfov_deg / 2.0))
    out = []
    for x, y, z, tid in poses:
        if x <= 0.05:
            continue                      # behind the camera
        nx = (-y / x) / half              # +y is left, +pixels are right
        ny = (z / x) / half * (image_w / max(image_h, 1))
        if abs(nx) > 1.0 or abs(ny) > 1.0:
            continue
        cx = (nx + 1.0) / 2.0 * image_w
        cy = (1.0 - ny) / 2.0 * image_h
        size_px = 2.0 * math.atan(object_size_m / (2.0 * x)) / math.radians(hfov_deg) * image_w
        out.append(
            Box(x=cx - size_px / 2, y=cy - size_px / 2, w=size_px, h=size_px,
                image_w=float(image_w), image_h=float(image_h), track_id=int(tid), label='truth')
        )
    return out
