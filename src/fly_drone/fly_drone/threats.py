"""
Throw things at the drone: the geometry and the Gazebo model behind the threat launcher.

A test of the escape reflex needs something to escape from, and the forest has nothing that
moves. :func:`aim` works out where to put an object so that it is ``distance_m`` from the
drone at a chosen bearing and elevation and flies straight at it (or past it, by ``miss_m``)
at ``speed_mps``; :func:`threat_sdf` describes the object as a static model; the launcher
node (:mod:`fly_drone.threat_launcher`) spawns it, moves it along its line every 50 ms of
simulation time through Gazebo's ``set_pose`` service, and removes it after its time of
flight. Kinematic, because the ``VelocityControl`` plugin on a spawned model was found inert
in gz-sim 10.5: the sphere spawned and sat there, and the first "reactions" measured were the
drone closing on a static ball.

The object has a visual and no collision: the lidar and the cameras see rendered geometry, so
it looms exactly like a bird, and if the brain fails to dodge it the drone is not knocked out
of the sky, which keeps a calibration sweep running.

This file has no rclpy in it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]

COLORS: dict[str, tuple[float, float, float]] = {
    'magenta': (1.0, 0.0, 1.0),
    'red': (1.0, 0.0, 0.0),
    'blue': (0.0, 0.0, 1.0),
    'yellow': (1.0, 1.0, 0.0),
    'green': (0.0, 1.0, 0.0),
    'white': (1.0, 1.0, 1.0),
    'black': (0.02, 0.02, 0.02),
}


@dataclass
class Launch:
    """One object on its way: what was spawned, where, and when to take it away again."""

    name: str
    position: Vec3
    velocity: Vec3
    radius_m: float
    color: str
    ttl_s: float
    """Seconds until the object is removed: the time of flight plus a margin."""
    distance_m: float
    bearing_deg: float
    elevation_deg: float
    speed_mps: float
    tof_s: float = 0.0
    """Time of flight to the intercept point, s."""
    t0: float = 0.0
    """Simulation time the object was spawned, s."""
    closest_m: float = float('inf')
    """Nearest the object's centre came to the drone's, so far."""
    closest_at: Vec3 = (0.0, 0.0, 0.0)
    """Where the object was at that moment."""
    closest_t: float = 0.0

    def note(self, position: Vec3, drone: Vec3, t: float) -> float:
        """Record this moment's separation; returns it."""
        d = math.sqrt(sum((position[i] - drone[i]) ** 2 for i in range(3)))
        if d < self.closest_m:
            self.closest_m, self.closest_at, self.closest_t = d, tuple(position), t
        return d

    def hit(self, drone_radius_m: float = 0.35) -> bool:
        """Did it touch the airframe: centres closer than the two radii."""
        return self.closest_m <= self.radius_m + drone_radius_m

    def position_at(self, t: float) -> Vec3:
        """Where the object is at simulation time ``t``: a straight line from the spawn."""
        dt = max(0.0, t - self.t0)
        return (
            self.position[0] + self.velocity[0] * dt,
            self.position[1] + self.velocity[1] * dt,
            self.position[2] + self.velocity[2] * dt,
        )


def _norm(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n < 1e-9:
        return (1.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def aim(
    drone_xyz: Vec3,
    yaw_rad: float,
    drone_vel_world: Vec3,
    distance_m: float,
    bearing_deg: float,
    elevation_deg: float,
    speed_mps: float,
    miss_m: float = 0.0,
    lead: bool = True,
) -> tuple[Vec3, Vec3, float]:
    """
    Where to spawn and which way to throw. Returns ``(position, velocity, time_of_flight)``.

    The spawn point is ``distance_m`` from the drone along ``bearing_deg`` (+ right of the
    nose) and ``elevation_deg`` (+ up), in the drone's body frame turned by its yaw. With
    ``lead`` the object is thrown at where the drone *will* be when they meet at its current
    velocity (a proper intercept), so a drone in level flight is still hit; ``miss_m`` moves
    the aim point that far to the drone's left (negative: right), for a near miss rather
    than a hit.
    """
    b, e = math.radians(bearing_deg), math.radians(elevation_deg)
    bx, by, bz = (
        distance_m * math.cos(e) * math.cos(b),
        -distance_m * math.cos(e) * math.sin(b),
        distance_m * math.sin(e),
    )
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    spawn = (drone_xyz[0] + c * bx - s * by, drone_xyz[1] + s * bx + c * by, drone_xyz[2] + bz)
    speed = max(speed_mps, 0.1)
    # Intercept: find T with |d + v T| = speed T, d the drone's offset from the spawn point
    # and v its velocity, so the object and the drone are at the same place at the same time.
    d = tuple(drone_xyz[i] - spawn[i] for i in range(3))
    v = drone_vel_world if lead else (0.0, 0.0, 0.0)
    tof = _intercept_time(d, v, speed)
    target = [drone_xyz[i] + v[i] * tof for i in range(3)]
    if miss_m:
        # perpendicular, in the horizontal plane, to the line of approach; + is the drone's
        # left when the object comes from ahead
        dx, dy = target[0] - spawn[0], target[1] - spawn[1]
        n = math.hypot(dx, dy)
        if n > 1e-6:
            target[0] += dy / n * miss_m
            target[1] += -dx / n * miss_m
    direction = _norm((target[0] - spawn[0], target[1] - spawn[1], target[2] - spawn[2]))
    velocity = (direction[0] * speed, direction[1] * speed, direction[2] * speed)
    return spawn, velocity, tof


def _intercept_time(d: Vec3, v: Vec3, speed: float) -> float:
    """Smallest positive T with |d + v T| = speed T; the straight flight time if there is none."""
    vv = sum(x * x for x in v)
    dv = sum(d[i] * v[i] for i in range(3))
    dd = sum(x * x for x in d)
    a, b, c = vv - speed * speed, 2.0 * dv, dd
    straight = math.sqrt(dd) / speed
    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            return straight
        t = -c / b
        return t if t > 0 else straight
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return straight
    roots = [(-b - math.sqrt(disc)) / (2.0 * a), (-b + math.sqrt(disc)) / (2.0 * a)]
    positive = [t for t in roots if t > 1e-6]
    return min(positive) if positive else straight


def marker_sdf(name: str, hit: bool, radius_m: float = 0.5) -> str:
    """A static, self-lit ball left where the sphere came closest: red for a hit, green for a miss."""
    r, g, b = (1.0, 0.1, 0.1) if hit else (0.1, 1.0, 0.1)
    return (
        f'<sdf version="1.9"><model name="{name}"><static>true</static><link name="body">'
        f'<visual name="v"><geometry><sphere><radius>{radius_m:.3f}</radius></sphere></geometry>'
        f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>'
        f'<emissive>{r * 0.6} {g * 0.6} {b * 0.6} 1</emissive></material></visual>'
        '</link></model></sdf>'
    )


def threat_sdf(name: str, radius_m: float, color: str) -> str:
    """A sphere with a visual and no collision, static so physics leaves it where it is put."""
    r, g, b = COLORS.get(color, COLORS['magenta'])
    return (
        f'<sdf version="1.9"><model name="{name}"><static>true</static>'
        '<link name="body">'
        f'<visual name="v"><geometry><sphere><radius>{radius_m:.3f}</radius></sphere></geometry>'
        f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>'
        '<specular>0.1 0.1 0.1 1</specular><emissive>0 0 0 1</emissive></material></visual>'
        '</link></model></sdf>'
    )
