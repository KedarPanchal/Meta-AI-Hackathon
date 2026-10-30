"""
The downward camera as a pinhole: footprint on the ground, and pixel <-> ground mapping.

Orientation, matching the ``camera_down`` sensor in the X3 model: the camera looks straight
down, image **up is the drone's forward** (+x body), image **right is the drone's right**
(-y body). Flat-ground assumption: the forest floor has about 6 m of relief, so a position
estimate from 15 m up is good to a few metres, which is what "go and look there" needs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

Point = tuple[float, float]


@dataclass(frozen=True)
class DownCamera:
    hfov_deg: float = 80.0
    width: int = 640
    height: int = 480

    @property
    def vfov_deg(self) -> float:
        half = math.tan(math.radians(self.hfov_deg / 2.0)) * self.height / self.width
        return 2.0 * math.degrees(math.atan(half))

    def half_extents(self, height_m: float) -> tuple[float, float]:
        """(half across-track, half along-track) ground extents at this height."""
        hx = height_m * math.tan(math.radians(self.hfov_deg / 2.0))
        hy = height_m * math.tan(math.radians(self.vfov_deg / 2.0))
        return hx, hy

    def footprint(self, height_m: float) -> tuple[float, float]:
        """(width across-track, length along-track) of the ground seen, m."""
        hx, hy = self.half_extents(height_m)
        return 2.0 * hx, 2.0 * hy

    def px_per_m(self, height_m: float) -> float:
        w, _ = self.footprint(height_m)
        return self.width / max(w, 1e-6)

    def pixel_to_ground(
        self, u: float, v: float, drone_xyz: tuple[float, float, float], yaw_rad: float,
        ground_z_m: float = 0.0,
    ) -> Point:
        """Normalised image coords (u right, v down, 0..1) to world x/y on flat ground."""
        px, py, pz = drone_xyz
        h = max(pz - ground_z_m, 0.1)
        hx, hy = self.half_extents(h)
        forward = (1.0 - 2.0 * v) * hy
        right = (2.0 * u - 1.0) * hx
        bx, by = forward, -right                    # body FLU
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        return (px + c * bx - s * by, py + s * bx + c * by)

    def ground_to_pixel(
        self, xy: Point, drone_xyz: tuple[float, float, float], yaw_rad: float,
        ground_z_m: float = 0.0, clip: bool = True,
    ) -> Point | None:
        """World x/y to normalised image coords; None if outside the frame unless ``clip`` is off."""
        px, py, pz = drone_xyz
        h = max(pz - ground_z_m, 0.1)
        hx, hy = self.half_extents(h)
        dx, dy = xy[0] - px, xy[1] - py
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        bx = c * dx + s * dy                        # forward
        by = -s * dx + c * dy                       # left
        forward, right = bx, -by
        u = (right / hx + 1.0) / 2.0
        v = (1.0 - forward / hy) / 2.0
        if not clip or (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return (u, v)
        return None
