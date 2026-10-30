"""
Spikes -> velocity. A priority state machine with a safety clamp on the way out.

Priority, highest first::

    ESCAPE > COOLDOWN > AVOID > NAVIGATE > PURSUE > HOLD

NAVIGATE is the mission layer's way in. It only ever sets a *goal position*; this layer turns
the position error into body-frame velocity, and because it sits under the reflexes, an
obstacle on the way produces an escape or a side-step and then the drone resumes closing on
the goal. The mission layer never sees velocity and never needs to.

The one idea worth stating plainly: **a giant-fibre spike is an event, not a rate.** DNp01
fires one or two spikes to a loom and then adapts, so the escape cannot be a feedback loop on
a spike rate. It is a triggered primitive: one spike commits the drone to a fixed manoeuvre
for ``hold_ms``, which then decays and is followed by a refractory ``cooldown_ms``. Treating
the spike count as a speed would make the drone stop escaping exactly when the threat got
closest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

from .config import Settings, settings
from .schemas import (
    BrainOut,
    Command,
    DroneState,
    Goal,
    GoalStatus,
    NavParams,
    Percept,
    PursuitParams,
    ReflexEvent,
    ReflexParams,
)

THREAT_ABOVE_DEG = 40.0
"""Above this elevation the threat is overhead, so the escape dives instead of climbing."""

AVOID_SIZE_DEG = 12.0
"""A loom this big with no giant-fibre spike still gets the graded side-step."""


@dataclass
class MotorLayer:
    """Holds the escape clock and the event log across ticks."""

    cfg: Settings = field(default_factory=settings)
    reflex: ReflexParams = field(default_factory=ReflexParams)
    pursuit: PursuitParams = field(default_factory=PursuitParams)
    nav: NavParams = field(default_factory=NavParams)
    events: list[ReflexEvent] = field(default_factory=list)
    max_events: int = 200

    # the mission layer's goal
    goal: Goal | None = None
    goal_reached: bool = False
    _goal_set_at: float | None = None
    _goal_reached_at: float | None = None

    # escape clock
    _escape_until: float = 0.0
    _decay_until: float = 0.0
    _cooldown_until: float = 0.0
    _escape_cmd: Command | None = None
    _avoiding: bool = False

    def __post_init__(self) -> None:
        # NAV_SPEED_MPS / NAV_ARRIVE_RADIUS_M apply to the default params only; an explicit
        # NavParams passed by a caller is taken as is.
        if self.nav == NavParams():
            self.nav.speed_mps = self.cfg.nav_speed_mps
            self.nav.arrive_radius_m = self.cfg.nav_arrive_radius_m

    # --- the one call --------------------------------------------------------

    def decide(
        self,
        percept: Percept,
        out: BrainOut,
        state: DroneState | None = None,
        now: float | None = None,
    ) -> Command:
        """Choose this tick's command from the brain's spikes and the percept."""
        now = time.monotonic() if now is None else now
        gf = int(out.spikes.get('gf', 0))
        ttmn = int(out.spikes.get('jump', 0))
        psi = int(out.spikes.get('flight_power', 0))

        triggered = gf > 0 or (self.reflex.accept_ttmn and ttmn > 0)
        if self.reflex.suppress_below_horizon and percept.elevation_deg < -5.0:
            # The ground expands exactly like a loom while descending.
            triggered = False

        # 1. a new trigger, if the reflex is armed and not refractory
        if triggered and self.reflex.enabled and now >= self._cooldown_until:
            cmd = self._begin_escape(percept, out, gf, ttmn, psi, now)
            return self._clamp(cmd, percept, state)

        # 2. an escape already in flight
        if now < self._decay_until and self._escape_cmd is not None:
            scale = 1.0
            if now > self._escape_until:
                span = max(self._decay_until - self._escape_until, 1e-6)
                scale = max(0.0, 1.0 - (now - self._escape_until) / span)
            e = self._escape_cmd
            cmd = Command(
                vx=e.vx * scale,
                vy=e.vy * scale,
                vz=e.vz * scale,
                yaw_rate=e.yaw_rate * scale,
                state='ESCAPE',
                reason=f"escape {'hold' if scale >= 1.0 else f'decay {scale:.2f}'}",
            )
            return self._clamp(cmd, percept, state)

        # 3. refractory: pursuit is allowed but damped
        damping = 1.0
        in_cooldown = now < self._cooldown_until
        if in_cooldown:
            damping = self.reflex.pursuit_damping

        # 4. a loom that did not reach threshold: graded side-step.
        #    Skipped while refractory: COOLDOWN outranks AVOID, or every escape would be
        #    followed by a side-step chasing the same receding object.
        if (
            self.reflex.enabled
            and not in_cooldown
            and percept.has_target
            and percept.size_deg >= AVOID_SIZE_DEG
            and percept.expansion_dps > 0.0
            and not (self.reflex.suppress_below_horizon and percept.elevation_deg < -5.0)
        ):
            cmd = self._avoid(percept)
            if not self._avoiding:      # one event per episode, not one per tick
                self._log(ReflexEvent(
                    t=time.time(), kind='AVOID', bearing_deg=percept.bearing_deg,
                    size_deg=percept.size_deg, expansion_dps=percept.expansion_dps,
                    distance_m=percept.distance_m, command=cmd,
                ))
            self._avoiding = True
            return self._clamp(cmd, percept, state)
        self._avoiding = False

        # 5. navigation to the mission layer's goal, damped while refractory
        if self.nav.enabled and self.goal is not None:
            if state is None or not state.connected:
                return self._clamp(
                    Command(state='HOLD', reason='goal set but no odometry yet'), percept, state
                )
            cmd = self._navigate(state, damping)
            if in_cooldown:
                cmd.reason += ' (damped: cooldown)'
            return self._clamp(cmd, percept, state)

        # 6. pursuit
        if self.pursuit.enabled and self.pursuit.arousal > 0.0:
            cmd = self._pursue(percept, out, damping)
            if in_cooldown:
                cmd.reason += ' (damped: cooldown)'
            return self._clamp(cmd, percept, state)

        # 7. nothing to do: hold station
        return self._clamp(
            Command(state='COOLDOWN' if in_cooldown else 'HOLD',
                    reason='refractory after escape' if in_cooldown else 'no drive'),
            percept, state,
        )

    # --- the behaviours ------------------------------------------------------

    def _begin_escape(
        self, percept: Percept, out: BrainOut, gf: int, ttmn: int, psi: int, now: float
    ) -> Command:
        r = self.reflex
        speed = r.speed_mps * (1.0 + r.per_extra_spike * max(0, gf - 1))
        if psi > 0:
            speed *= 1.0 + r.psi_bonus

        bearing = percept.bearing_deg
        if r.direction_source == 'brain':
            # Left/right giant-fibre imbalance. On the toy graph this points TOWARD the threat,
            # because loom input reaches both giant fibres; 'vision' is the default for a reason.
            gl, gr = out.spikes_lr.get('gf', (0, 0))
            if gl or gr:
                bearing = 90.0 * (gr - gl) / max(1, gl + gr)

        # Away from the threat, in body FLU. A threat at bearing b lies along
        # (cos b, -sin b) in (forward, left), so the escape runs along (-cos b, +sin b).
        climb = r.climb_deg
        if percept.elevation_deg > THREAT_ABOVE_DEG:
            climb = -abs(climb)          # threat overhead: dive and slide sideways instead
        b = math.radians(bearing)
        ch, cv = math.cos(math.radians(climb)), math.sin(math.radians(climb))
        cmd = Command(
            vx=-math.cos(b) * ch * speed,
            vy=math.sin(b) * ch * speed,
            vz=cv * speed,
            yaw_rate=0.0,
            state='ESCAPE',
            reason=f'GF x{gf}' + (f' +TTMn x{ttmn}' if ttmn else '') + (' +PSI' if psi else '')
            + f', away from {bearing:+.0f} deg, {climb:+.0f} deg pitch, {speed:.1f} m/s',
        )
        self._escape_cmd = cmd
        self._escape_until = now + r.hold_ms / 1000.0
        self._decay_until = self._escape_until + r.decay_ms / 1000.0
        self._cooldown_until = self._decay_until + r.cooldown_ms / 1000.0
        ttc = None
        if percept.distance_m is not None and percept.expansion_dps > 0 and percept.size_deg > 0:
            # tau = theta / theta_dot, the optical time-to-contact the fly itself uses
            ttc = percept.size_deg / percept.expansion_dps
        self._log(
            ReflexEvent(
                t=time.time(), kind='ESCAPE', gf_spikes=gf, ttmn_spikes=ttmn, psi_spikes=psi,
                bearing_deg=percept.bearing_deg, size_deg=percept.size_deg,
                expansion_dps=percept.expansion_dps, distance_m=percept.distance_m,
                ttc_s=ttc, command=cmd,
            )
        )
        return cmd

    def _avoid(self, percept: Percept) -> Command:
        """Graded side-step: the fly's long-mode takeoff, scaled by how big the thing is."""
        urgency = min(1.0, percept.size_deg / 45.0)
        speed = self.reflex.avoid_speed_mps * urgency
        b = math.radians(percept.bearing_deg)
        return Command(
            vy=math.sin(b) * speed if abs(math.sin(b)) > 1e-3 else speed,
            vx=-math.cos(b) * speed * 0.3,
            state='AVOID',
            reason=f'loom {percept.size_deg:.0f} deg, no GF spike -> side-step {speed:.2f} m/s',
        )

    def _pursue(self, percept: Percept, out: BrainOut, damping: float) -> Command:
        """Descending-neuron rates to (forward, yaw). Gain-only forward, so arousal=0 hovers."""
        p = self.pursuit
        sl, sr = out.spikes_lr.get('steer', (0, 0))
        fwd_spikes = int(out.spikes.get('forward', 0))
        yaw = (p.yaw_gain * (sl - sr) + p.yaw_bias) * damping
        vx = p.forward_gain * fwd_spikes * damping
        if percept.distance_m is not None and percept.distance_m < p.stop_distance_m:
            vx = 0.0
        reason = f'steer L{sl}/R{sr}, forward {fwd_spikes}'
        if not p.yaw_gain and not p.forward_gain:
            reason += ' (decoder not calibrated: outputs.steer/forward are empty)'
        return Command(vx=vx, yaw_rate=yaw, state='PURSUE', reason=reason)

    def _navigate(self, state: DroneState, damping: float) -> Command:
        """
        Position error to body-frame velocity: a saturated P controller with three rules.

        * **Climb first.** Well below the goal, horizontal speed is zero until the drone is
          within ``climb_first_dz_m`` of the flight level. The forest is crossed above the
          trees or not at all.
        * **Face the travel direction** so the forward camera, and the loom reflex behind it,
          looks where the drone is going. Translation does not wait for the yaw: the body-frame
          velocity is computed from the current heading, so the path is straight either way.
        * **Arrival latches.** Once inside the arrival box the goal stays reached, and the
          same controller keeps station there, so a gust does not un-reach a waypoint.
        """
        g, n = self.goal, self.nav
        assert g is not None
        px, py, pz = state.position
        dx, dy, dz = g.x - px, g.y - py, g.z - pz
        horiz = math.hypot(dx, dy)
        radius = g.arrive_radius_m if g.arrive_radius_m is not None else n.arrive_radius_m
        if not self.goal_reached and horiz <= radius and abs(dz) <= n.arrive_dz_m:
            self.goal_reached = True
            self._goal_reached_at = time.time()

        cap = g.speed_mps if g.speed_mps is not None else n.speed_mps
        speed = min(n.kp * horiz, cap) * damping
        climbing_first = dz > n.climb_first_dz_m
        if climbing_first:
            speed = 0.0
        if horiz > 1e-6:
            vwx, vwy = dx / horiz * speed, dy / horiz * speed
        else:
            vwx = vwy = 0.0
        # world (ENU) velocity into body FLU: x forward along the heading, y to its left
        c, s = math.cos(state.yaw_rad), math.sin(state.yaw_rad)
        vx = c * vwx + s * vwy
        vy = -s * vwx + c * vwy
        vz = n.altitude_gain * dz

        yaw_rate = 0.0
        if n.face_travel and speed > n.heading_min_speed_mps:
            want = math.atan2(vwy, vwx)
            err = (want - state.yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
            yaw_rate = n.yaw_gain * err

        name = g.label or g.id or f'({g.x:.0f}, {g.y:.0f}, {g.z:.0f})'
        if self.goal_reached:
            reason = f'at goal {name}, station-keeping ({horiz:.1f} m off)'
        elif climbing_first:
            reason = f'climbing to z={g.z:.1f} m before moving to {name} ({dz:+.1f} m to go)'
        else:
            reason = f'to goal {name}: {horiz:.1f} m, dz {dz:+.1f} m, {speed:.1f} m/s'
        return Command(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate, state='NAVIGATE', reason=reason)

    # --- the goal ------------------------------------------------------------

    def set_goal(self, goal: Goal) -> None:
        """Replace the goal. Arrival state starts over; the escape clock is untouched."""
        self.goal = goal
        self.goal_reached = False
        self._goal_set_at = time.time()
        self._goal_reached_at = None

    def clear_goal(self) -> None:
        self.goal = None
        self.goal_reached = False
        self._goal_set_at = None
        self._goal_reached_at = None

    def goal_status(self, state: DroneState) -> GoalStatus:
        g = self.goal
        st = GoalStatus(
            active=g is not None,
            reached=self.goal_reached,
            goal=g,
            state=state.state,
            position=state.position,
            yaw_rad=state.yaw_rad,
            connected=state.connected,
            output_enabled=state.output_enabled,
            set_at=self._goal_set_at,
            reached_at=self._goal_reached_at,
            ticks=state.ticks,
        )
        if g is not None:
            px, py, pz = state.position
            dx, dy, dz = g.x - px, g.y - py, g.z - pz
            st.horizontal_m = math.hypot(dx, dy)
            st.dz_m = dz
            st.distance_m = math.sqrt(dx * dx + dy * dy + dz * dz)
            cap = g.speed_mps if g.speed_mps is not None else self.nav.speed_mps
            if cap > 0:
                st.eta_s = st.horizontal_m / cap + abs(dz) / max(self.cfg.max_climb, 0.1)
        return st

    # --- safety --------------------------------------------------------------

    def _clamp(self, cmd: Command, percept: Percept, state: DroneState | None) -> Command:
        """Apply the safety envelope. Every command ends here and nothing bypasses it."""
        c = self.cfg
        alt = percept.altitude_m if percept.altitude_m else (state.altitude_m if state else 0.0)
        # altitude hold, unless the escape is already commanding vertical motion. With a goal
        # set the level to hold is the goal's, not the pursuit cruise altitude: a side-step at
        # 15 m must not also start a descent to 3 m.
        hold_alt = self.goal.z if self.goal is not None else self.pursuit.cruise_altitude_m
        if cmd.state in ('HOLD', 'PURSUE', 'AVOID') and hold_alt > 0:
            err = hold_alt - alt
            cmd.vz = self.pursuit.altitude_gain * err
        # hard floor: never command descent below the minimum
        if alt and alt < c.min_altitude_m and cmd.vz < 0:
            cmd.vz = 0.0
            cmd.reason += f' [floor {c.min_altitude_m} m]'
        horiz = math.hypot(cmd.vx, cmd.vy)
        if horiz > c.max_speed:
            s = c.max_speed / horiz
            cmd.vx *= s
            cmd.vy *= s
            cmd.reason += f' [clamped to {c.max_speed} m/s]'
        cmd.vz = max(-c.max_climb, min(c.max_climb, cmd.vz))
        cmd.yaw_rate = max(-c.max_yaw, min(c.max_yaw, cmd.yaw_rate))
        return cmd

    # --- bookkeeping ---------------------------------------------------------

    def _log(self, event: ReflexEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            del self.events[: len(self.events) - self.max_events]

    def state_name(self, now: float | None = None) -> str:
        now = time.monotonic() if now is None else now
        if now < self._decay_until:
            return 'ESCAPE'
        if now < self._cooldown_until:
            return 'COOLDOWN'
        if self.goal is not None:
            return 'NAVIGATE'
        return 'HOLD'

    def reset(self) -> None:
        self._escape_until = self._decay_until = self._cooldown_until = 0.0
        self._escape_cmd = None
