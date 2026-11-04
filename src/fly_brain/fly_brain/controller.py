"""
The single shared controller: one brain, one motor layer, one lock.

FastAPI, the MCP tools and the ROS node all run their handlers on worker threads and all talk
to the same object, so every mutation here is under one reentrant lock.

The brain loads lazily because ``FastMCP.from_fastapi`` drives the app in-process over an
httpx ASGI transport, which never runs FastAPI's startup or lifespan events. Anything that
waited for startup would never happen, so :meth:`Controller.ensure_brain` builds the network
on first use instead. The local connectome runtime is pure NumPy; there is no Brian2 import
or worker-thread SIGINT trap.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from .brain import make_brain
from .bridge import Bridge, NullBridge
from .circuit import Circuit
from .config import Settings, settings
from .encoders import EncoderParams, encode
from .motor import MotorLayer
from .perception import AudioFrontEnd, LoomTracker
from .schemas import Box, BrainOut, BrainStatus, Command, DroneState, Goal, GoalStatus, Percept


class Controller:
    """Owns the brain, the decoders and the last known vehicle state."""

    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or settings()
        self.lock = threading.RLock()
        self.circuit = Circuit(path=self.cfg.circuit_path or None)
        self.encoder = EncoderParams()
        self.motor = MotorLayer(cfg=self.cfg)
        self.tracker = LoomTracker(hfov_deg=self.cfg.camera_hfov_deg)
        self.audio = AudioFrontEnd()
        self.bridge: Bridge = NullBridge()
        self.state = DroneState()
        self.brain: Any = None
        self.output_enabled = False
        self.last_percept = Percept()
        self.last_out: BrainOut | None = None
        self.load_error: str | None = None

    # --- brain lifecycle -----------------------------------------------------

    def ensure_brain(self, backend: str | None = None) -> Any:
        """Build the brain on first use. Safe to call from any thread."""
        with self.lock:
            if self.brain is None or (backend and backend != self.brain.backend):
                self.load_brain(backend=backend)
            return self.brain

    def load_brain(
        self,
        backend: str | None = None,
        ablate: list[str] | None = None,
        prune_hops: int | None = None,
    ) -> BrainStatus:
        """(Re)build the network. Replaces any brain already loaded."""
        with self.lock:
            backend = backend or self.cfg.backend
            try:
                self.brain = make_brain(
                    backend=backend,
                    circuit=self.circuit,
                    cfg=self.cfg,
                    ablate=ablate,
                    prune_hops=prune_hops,
                )
                self.load_error = None
            except Exception as exc:               # surfaced through brain_status, not swallowed
                self.brain = None
                self.load_error = f'{type(exc).__name__}: {exc}'
                raise
            self.motor.reset()
            return self.brain.status()

    def status(self) -> BrainStatus:
        with self.lock:
            if self.brain is None:
                return BrainStatus(
                    backend=self.cfg.backend,
                    loaded=False,
                    data_dir=str(self.cfg.connectome_file()),
                    tick_ms=self.cfg.brain_tick_ms,
                    missing_types=[] if not self.load_error else [self.load_error],
                )
            return self.brain.status()

    # --- the control loop ----------------------------------------------------

    def tick(
        self, percept: Percept, send: bool | None = None, now: float | None = None
    ) -> Command:
        """
        One full pass: encode -> brain -> motor -> bridge.

        The command is computed whatever happens; it only reaches the vehicle when output is
        enabled, so the whole stack can be exercised against a live sim with nothing moving.

        ``now`` is the clock the escape timing (hold, decay, cooldown) runs on. The ROS node
        passes its own clock so that, under ``use_sim_time``, an escape lasts its 350 ms of
        *simulated* time however slowly the simulator renders. Default: wall clock.
        """
        with self.lock:
            brain = self.ensure_brain()
            # A Muse-selected investigate goal raises the reproductive gate. The
            # goal is still only a position; the pC1 -> DNp13 readout modulates
            # approach commitment in MotorLayer, never the heading.
            if self.motor.goal is not None and self.motor.goal.arousal > 0.0:
                percept = percept.model_copy(
                    update={'arousal': max(percept.arousal, self.motor.goal.arousal)}
                )
            drive = encode(percept, self.encoder)
            out = brain.step(drive, self.cfg.brain_tick_ms)
            last = self.tracker.last
            cmd = self.motor.decide(
                percept, out, self.state, now=time.monotonic() if now is None else now,
                track_key=f'{last.source}:{last.track_id}' if last is not None else '',
            )
            self.last_percept = percept
            self.last_out = out
            self.state.state = cmd.state
            self.state.last_command = cmd
            self.state.ticks += 1
            self.state.output_enabled = self.output_enabled
            if (self.output_enabled if send is None else send):
                self.bridge.send(cmd)
            return cmd

    def tick_from_boxes(
        self, boxes: list[Box], t: float | None = None, **percept_kw: Any
    ) -> Command:
        """Run one tick straight from detections: the convenience path for the ROS node."""
        with self.lock:
            p = self.tracker.update(boxes, t if t is not None else time.monotonic())
            for k, v in percept_kw.items():
                setattr(p, k, v)
            p.altitude_m = percept_kw.get('altitude_m', self.state.altitude_m)
            p.arousal = percept_kw.get('arousal', self.motor.pursuit.arousal)
            return self.tick(p)

    # --- output gate ---------------------------------------------------------

    def enable_output(self, enabled: bool) -> bool:
        """
        Open or close the arming switch. Disabling it also sends one stop command.

        Arming also calls the bridge's ``arm()`` if it has one: Gazebo's multicopter controller
        wants an explicit enable before it will spin the rotors.
        """
        with self.lock:
            self.output_enabled = bool(enabled)
            self.state.output_enabled = self.output_enabled
            if enabled:
                arm = getattr(self.bridge, 'arm', None)
                if callable(arm):
                    arm()
            else:
                self.bridge.stop()
            return self.output_enabled

    def stop(self) -> Command:
        """Zero velocity, disarm output, drop the goal, clear the escape clock."""
        with self.lock:
            self.output_enabled = False
            self.state.output_enabled = False
            self.motor.reset()
            self.motor.clear_goal()
            cmd = Command(state='HOLD', reason='stop requested')
            self.bridge.stop()
            self.state.last_command = cmd
            self.state.state = 'HOLD'
            return cmd

    def set_bridge(self, bridge: Bridge) -> str:
        with self.lock:
            self.bridge = bridge
            return bridge.name

    def set_dodge_advice(self, advice) -> None:
        """Muse's (or anyone's) opinion on which way to dodge the current threat."""
        with self.lock:
            self.motor.set_advice(advice)

    # --- the goal: the mission layer's only lever -----------------------------

    def set_goal(self, goal: Goal) -> GoalStatus:
        """Point the brain at a world position. The next tick starts closing on it."""
        with self.lock:
            self.motor.set_goal(goal)
            return self.goal_status()

    def clear_goal(self) -> GoalStatus:
        """Forget the goal; the brain falls back to HOLD at the pursuit cruise altitude."""
        with self.lock:
            self.motor.clear_goal()
            return self.goal_status()

    def goal_status(self) -> GoalStatus:
        with self.lock:
            self.state.output_enabled = self.output_enabled
            return self.motor.goal_status(self.state)

    # --- experiments ---------------------------------------------------------

    def stimulate(self, types: list[str], hz: float = 20.0, ms: float = 200.0) -> dict:
        """
        Drive arbitrary types on a **separate model copy** and report what spiked.

        A separate copy for two reasons: it must not disturb the flying controller's network,
        and CNSModel replaces stimulated types with Poisson sources, so asking about a type
        here would otherwise delete it from the live brain for good.
        """
        with self.lock:
            brain = self.brain
        if brain is None or not hasattr(brain, 'stimulate'):
            return {'backend': self.cfg.backend, 'note': 'the brain is not loaded', 'spikes': {}}
        return brain.stimulate(types, hz=hz, ms=ms)


_controller: Controller | None = None
_controller_lock = threading.Lock()


def controller() -> Controller:
    """Process-wide controller singleton."""
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = Controller()
        return _controller


def reset_controller() -> None:
    """Drop the singleton. For tests only."""
    global _controller
    with _controller_lock:
        _controller = None
