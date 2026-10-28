"""Tuning and characterising the escape reflex. Four tools."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Query
from pydantic import BaseModel, Field

from fly_brain.controller import controller
from fly_brain.perception import approach_percepts
from fly_brain.schemas import PursuitParams, ReflexParams

router = APIRouter(prefix="/reflex", tags=["reflex"])


class MotorConfigPatch(BaseModel):
    reflex: dict = Field(
        default_factory=dict, description="Fields of ReflexParams to change, by name."
    )
    pursuit: dict = Field(
        default_factory=dict, description="Fields of PursuitParams to change, by name."
    )
    encoder_gains: dict[str, float] = Field(
        default_factory=dict,
        description="Per-channel input gain. The loom gain is the knob that sets how early "
        "the reflex fires; test_reflex measures the effect.",
    )


class TestRequest(BaseModel):
    distance_m: float = Field(default=5.0, gt=0, le=50, description="Starting range of the object.")
    speed_mps: float = Field(default=5.0, gt=0, le=50, description="Closing speed.")
    object_size_m: float = Field(default=0.5, gt=0, le=20, description="Physical diameter.")
    bearing_deg: float = Field(default=0.0, ge=-90, le=90, description="Bearing; + is RIGHT.")
    sound_db: float = Field(
        default=-80.0,
        description="Simultaneous sound level, dBFS. FLYCNS reports that on the real "
        "connectome sound LOWERS the giant-fibre loom response, while on the toy graph it "
        "raises it, so this flips sign between datasets.",
    )
    altitude_m: float = Field(default=3.0, ge=0, description="Altitude during the test.")


@router.get("/config", operation_id="get_motor_config")
def get_motor_config() -> dict:
    """Current reflex, pursuit and input-gain settings, with the safety envelope."""
    ctl = controller()
    with ctl.lock:
        return {
            "reflex": ctl.motor.reflex.model_dump(),
            "pursuit": ctl.motor.pursuit.model_dump(),
            "encoder": {
                "peak_hz": ctl.encoder.peak_hz,
                "gains": dict(ctl.encoder.gains),
                "expansion_sat_dps": ctl.encoder.expansion_sat_dps,
                "size_sat_deg": ctl.encoder.size_sat_deg,
                "target_sat_deg": ctl.encoder.target_sat_deg,
            },
            "safety": {
                "max_speed": ctl.cfg.max_speed,
                "max_climb": ctl.cfg.max_climb,
                "max_yaw": ctl.cfg.max_yaw,
                "min_altitude_m": ctl.cfg.min_altitude_m,
            },
        }


@router.patch("/config", operation_id="update_motor_config")
def update_motor_config(patch: Annotated[MotorConfigPatch, Body()] = MotorConfigPatch()) -> dict:
    """Change reflex timing, pursuit gains or input gains, live.

    Unknown field names are reported rather than ignored, so a typo in a tuning session does
    not look like a setting that had no effect.
    """
    patch = patch or MotorConfigPatch()
    ctl = controller()
    unknown: list[str] = []
    with ctl.lock:
        for key, value in patch.reflex.items():
            if key in ReflexParams.model_fields:
                setattr(ctl.motor.reflex, key, value)
            else:
                unknown.append(f"reflex.{key}")
        for key, value in patch.pursuit.items():
            if key in PursuitParams.model_fields:
                setattr(ctl.motor.pursuit, key, value)
            else:
                unknown.append(f"pursuit.{key}")
        for key, value in patch.encoder_gains.items():
            if key in ctl.encoder.gains:
                ctl.encoder.gains[key] = float(value)
            else:
                unknown.append(f"encoder_gains.{key}")
        return {
            "reflex": ctl.motor.reflex.model_dump(),
            "pursuit": ctl.motor.pursuit.model_dump(),
            "encoder_gains": dict(ctl.encoder.gains),
            "unknown_fields": unknown,
        }


@router.get("/events", operation_id="reflex_events")
def reflex_events(
    limit: Annotated[int, Query(ge=1, le=200, description="Most recent events to return.")] = 20,
) -> dict:
    """The log of escapes and side-steps: what fired, when, and how close the threat was."""
    ctl = controller()
    with ctl.lock:
        events = [e.model_dump() for e in ctl.motor.events[-limit:]]
        return {"n_total": len(ctl.motor.events), "events": events}


@router.post("/test", operation_id="test_reflex")
def test_reflex(req: Annotated[TestRequest, Body()] = TestRequest()) -> dict:
    """Replay a synthetic approaching object and report exactly when the reflex fired.

    This is the calibration loop. The trigger has to satisfy

        tau_trigger >= latency + sqrt(2 * clearance / acceleration)

    which is about 0.75 s for 0.9 m of clearance at 5 m/s^2 with 0.15 s of latency. If
    ``ttc_at_trigger_s`` comes back below that, raise the loom gain with
    ``update_motor_config`` and run this again.

    **Output is forced off for the duration**, whatever the arming state, and restored
    afterwards: this flies a fictional object past a real aircraft.
    """
    req = req or TestRequest()
    ctl = controller()
    seq = approach_percepts(
        distance_m=req.distance_m,
        speed_mps=req.speed_mps,
        object_size_m=req.object_size_m,
        bearing_deg=req.bearing_deg,
        dt=ctl.cfg.brain_tick_ms / 1000.0,
        altitude_m=req.altitude_m,
        sound_db=req.sound_db,
    )
    with ctl.lock:
        was_enabled = ctl.output_enabled
        n_events = len(ctl.motor.events)
        ctl.output_enabled = False
        ctl.motor.reset()
    trace = []
    trigger = None
    try:
        for i, p in enumerate(seq):
            cmd = ctl.tick(p, send=False)
            out = ctl.last_out
            gf = int(out.spikes.get("gf", 0)) if out else 0
            t = i * ctl.cfg.brain_tick_ms / 1000.0
            row = {
                "t_s": round(t, 3),
                "distance_m": round(p.distance_m or 0.0, 3),
                "size_deg": round(p.size_deg, 2),
                "expansion_dps": round(p.expansion_dps, 1),
                "gf_spikes": gf,
                "state": cmd.state,
            }
            trace.append(row)
            if trigger is None and cmd.state == "ESCAPE":
                trigger = {
                    "t_s": round(t, 3),
                    "distance_m": round(p.distance_m or 0.0, 3),
                    "ttc_at_trigger_s": round((p.distance_m or 0.0) / req.speed_mps, 3),
                    "gf_spikes": gf,
                    "size_deg": round(p.size_deg, 2),
                    "expansion_dps": round(p.expansion_dps, 1),
                    "command": cmd.model_dump(),
                }
    finally:
        with ctl.lock:
            ctl.output_enabled = was_enabled
            ctl.motor.reset()

    required = 0.15 + (2.0 * 0.9 / 5.0) ** 0.5
    verdict = "NEVER FIRED"
    if trigger:
        verdict = (
            "in time" if trigger["ttc_at_trigger_s"] >= required else "TOO LATE: raise the loom gain"
        )
    return {
        "stimulus": req.model_dump(),
        "backend": ctl.status().backend,
        "loom_gain": ctl.encoder.gains.get("loom_speed"),
        "trigger": trigger,
        "required_ttc_s": round(required, 3),
        "verdict": verdict,
        "new_events": [e.model_dump() for e in ctl.motor.events[n_events:]],
        "trace": trace,
    }
