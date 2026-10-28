"""
Spikes -> velocity. A priority state machine with a safety clamp on the way out.

Priority, highest first::

    ESCAPE > COOLDOWN > AVOID > PURSUE > HOLD

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
    events: list[ReflexEvent] = field(default_factory=list)
    max_events: int = 200

    # escape clock
    _escape_until: float = 0.0
    _decay_until: float = 0.0
    _cooldown_until: float = 0.0
    _escape_cmd: Command | None = None
    _avoiding: bool = False

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

        # 5. pursuit
        if self.pursuit.enabled and self.pursuit.arousal > 0.0:
            cmd = self._pursue(percept, out, damping)
            if in_cooldown:
                cmd.reason += ' (damped: cooldown)'
            return self._clamp(cmd, percept, state)

        # 6. nothing to do: hold station
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

    # --- safety --------------------------------------------------------------

    def _clamp(self, cmd: Command, percept: Percept, state: DroneState | None) -> Command:
        """Apply the safety envelope. Every command ends here and nothing bypasses it."""
        c = self.cfg
        alt = percept.altitude_m if percept.altitude_m else (state.altitude_m if state else 0.0)
        # altitude hold, unless the escape is already commanding vertical motion
        if cmd.state in ('HOLD', 'PURSUE', 'AVOID') and self.pursuit.cruise_altitude_m > 0:
            err = self.pursuit.cruise_altitude_m - alt
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
        return 'HOLD'

    def reset(self) -> None:
        self._escape_until = self._decay_until = self._cooldown_until = 0.0
        self._escape_cmd = None
