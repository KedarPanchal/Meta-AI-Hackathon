"""
Which way to dodge: a prediction of where the object goes, and the way out of its path.

The old reflex always ran *away* from the threat's bearing and 35 degrees up. Against a thing
coming straight at the drone that is the worst of both worlds: a retreat along its line gains
nothing on an object faster than the aircraft, and climbing under it puts the airframe in its
path and, for a bigger object, in its downwash. A fly's own escape is a sideways jump. So the
escape now has a **dodge direction** decided from the object's predicted path:

* From the tracked relative position and velocity, the object's point of closest approach
  (``miss``) says which side it will pass on, if any. Dodge **away from that side**.
* Anything coming straight at the drone, or from steeply above or below, gets a **sideways**
  dodge: it leaves the object's line fastest, and never puts the drone under or over it.
* Only a level crossing from the side (bearing beyond 45 degrees) gets a **vertical** dodge,
  up by default, down if the object will pass above.

When Muse Spark has been asked (``fly_drone.dodge_advisor``) and answered in time for this
track, its advice wins; it has seen the frame and can tell a bird from a ball. The geometry
is the fallback that is always there in time.
"""
from __future__ import annotations

import math

from .schemas import DodgeAdvice, Percept

STEEP = 0.5
"""|v_z| / |v| above which the object is coming down on (or up at) the drone."""
SIDE_BEARING_DEG = 45.0
"""Beyond this bearing a level object is crossing from the side, not coming head-on."""
MISS_CLEAR_M = 1.5
"""A predicted miss smaller than this is a collision course for a 0.5 m object and a 0.5 m drone."""
DEFAULT_RANGE_M = 8.0
"""Range assumed for a camera-only threat, which measures none."""
DIRECTIONS = {
    'left': (0.0, 1.0, 0.0), 'right': (0.0, -1.0, 0.0),
    'up': (0.0, 0.0, 1.0), 'down': (0.0, 0.0, -1.0),
}


def relative_state(percept: Percept) -> tuple[tuple[float, float, float], tuple[float, float, float] | None]:
    """The object's position in body FLU from the percept's angles, and its velocity if known."""
    b, e = math.radians(percept.bearing_deg), math.radians(percept.elevation_deg)
    r = percept.distance_m if percept.distance_m else DEFAULT_RANGE_M
    p = (r * math.cos(e) * math.cos(b), -r * math.cos(e) * math.sin(b), r * math.sin(e))
    v = tuple(percept.velocity_body) if percept.velocity_body is not None else None
    return p, v


def closest_approach(p, v) -> tuple[tuple[float, float, float], float]:
    """The object's position relative to the drone when nearest, and when (s; 0 if receding)."""
    vv = sum(x * x for x in v)
    if vv < 1e-6:
        return tuple(p), 0.0
    t = max(0.0, -sum(p[i] * v[i] for i in range(3)) / vv)
    return tuple(p[i] + v[i] * t for i in range(3)), t


def choose_dodge(
    percept: Percept,
    prefer: str = 'auto',
    advice: DodgeAdvice | None = None,
    track_key: str = '',
    now: float | None = None,
    advice_max_age_s: float = 6.0,
) -> DodgeAdvice:
    """Decide the dodge for this percept: Muse's advice if fresh and about this track, else geometry."""
    if advice is not None and advice.confidence >= 0.5:
        fresh = now is None or advice.t <= 0.0 or now - advice.t <= advice_max_age_s
        about = not advice.track_key or not track_key or advice.track_key == track_key
        if fresh and about and (prefer == 'auto' or prefer == advice.axis):
            return advice
    p, v = relative_state(percept)
    speed = math.sqrt(sum(x * x for x in v)) if v is not None else 0.0
    if v is None or speed < 0.5:
        # unknown or static-looking: assume it flies straight at us along its line of sight
        r = math.sqrt(sum(x * x for x in p)) or 1.0
        v = tuple(-5.0 * x / r for x in p)
        speed = 5.0
        expected = 'straight at us (no velocity measured)'
    else:
        expected = ''
    miss, t_close = closest_approach(p, v)
    miss_lat, miss_vert = miss[1], miss[2]
    steep = abs(v[2]) / speed > STEEP
    overhead = abs(percept.elevation_deg) > 40.0
    from_side = abs(percept.bearing_deg) > SIDE_BEARING_DEG and not steep
    on_course = math.sqrt(miss_lat ** 2 + miss_vert ** 2) < MISS_CLEAR_M
    if not expected:
        if not on_course:
            side = ('to our left' if miss_lat > 0 else 'to our right') if abs(miss_lat) >= abs(miss_vert) else ('above us' if miss_vert > 0 else 'below us')
            expected = f'passing {side} by {math.hypot(miss_lat, miss_vert):.1f} m'
        elif steep:
            expected = 'coming down on us' if v[2] < 0 else 'coming up at us'
        else:
            expected = 'straight at us, level'

    axis = prefer if prefer in ('lateral', 'vertical') else (
        'vertical' if from_side and not overhead else 'lateral'
    )
    if axis == 'lateral':
        # away from the side it will pass on; with no side, away from its lateral drift; else left
        if abs(miss_lat) > 0.15:
            direction = 'right' if miss_lat > 0 else 'left'
        elif abs(v[1]) > 0.3:
            direction = 'right' if v[1] > 0 else 'left'
        else:
            direction = 'left'
        reason = 'sideways: ' + ('it comes from above/below' if steep or overhead else
                                 'it comes head-on' if not from_side else 'forced lateral')
    else:
        direction = 'down' if miss_vert > 0.3 else 'up'
        reason = 'vertical: a level crossing from the side'
    return DodgeAdvice(
        axis=axis, direction=direction, expected=expected, confidence=0.6 if percept.velocity_body else 0.4,
        source='geometry' if percept.velocity_body else 'default', reason=reason, track_key=track_key,
        t=now or 0.0,
    )


def escape_vector(p, dodge: DodgeAdvice, away_fraction: float, climb_deg: float) -> tuple[float, float, float]:
    """Unit escape direction in body FLU: the dodge, plus a push straight away from the object.

    A vertical dodge is pitched ``climb_deg`` above (or below) horizontal rather than straight
    up, as the old reflex was: it still gets out of the line while keeping airspeed.
    """
    d = DIRECTIONS[dodge.direction]
    r = math.sqrt(sum(x * x for x in p)) or 1.0
    away = tuple(-x / r for x in p)
    if dodge.axis == 'vertical':
        c, s = math.cos(math.radians(climb_deg)), math.sin(math.radians(climb_deg))
        horiz = math.hypot(away[0], away[1]) or 1.0
        d = (away[0] / horiz * c, away[1] / horiz * c, s * (1.0 if dodge.direction == 'up' else -1.0))
        away = (0.0, 0.0, 0.0)
    vec = tuple(d[i] + away_fraction * away[i] for i in range(3))
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return tuple(x / n for x in vec)
