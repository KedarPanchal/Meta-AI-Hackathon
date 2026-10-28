"""Flying it. Four tools, one of which is the arming switch."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from fly_brain.controller import controller
from fly_brain.schemas import DroneState

router = APIRouter(prefix="/drone", tags=["drone"])


class OutputRequest(BaseModel):
    enabled: bool = Field(
        description="True lets computed commands reach the vehicle. The whole stack runs "
        "either way; this only opens the gate at the very end."
    )


class PursuitRequest(BaseModel):
    arousal: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="The P1/pC1 courtship gate, 0..1. This is a STATE, not a direction: in "
        "FLYCNS these cells drive the song command neurons, not locomotion, so they work as "
        "'pursue now' while the heading comes from vision. At 0 the drone hovers.",
    )
    yaw_gain: float | None = Field(default=None, description="rad/s per left-right spike imbalance.")
    forward_gain: float | None = Field(
        default=None,
        description="m/s per forward-neuron spike. Fitted WITHOUT an intercept on purpose: an "
        "intercept let the drone creep forward with the gate closed.",
    )
    stop_distance_m: float | None = Field(default=None, description="Stop closing inside this range.")
    cruise_altitude_m: float | None = Field(default=None, description="Altitude to hold, m.")
    enabled: bool | None = Field(default=None, description="Allow pursuit to command velocity at all.")


@router.get("/state", operation_id="drone_state", response_model=DroneState)
def drone_state() -> DroneState:
    """Last known vehicle state, the current behaviour, and whether output is armed."""
    ctl = controller()
    with ctl.lock:
        ctl.state.output_enabled = ctl.output_enabled
        ctl.state.state = ctl.motor.state_name() if ctl.state.last_command is None else ctl.state.state
        return ctl.state


@router.post("/output", operation_id="enable_output")
def enable_output(req: Annotated[OutputRequest, Body()]) -> dict:
    """Arm or disarm the vehicle output.

    Disarming also sends one zero-velocity command, so the aircraft does not coast on the
    last setpoint. Run the whole system disarmed first and watch ``drone_state``.
    """
    ctl = controller()
    enabled = ctl.enable_output(req.enabled)
    return {
        "output_enabled": enabled,
        "bridge": ctl.bridge.name,
        "note": "commands now reach the vehicle" if enabled else "commands are computed and dropped",
    }


@router.post("/pursuit", operation_id="set_pursuit")
def set_pursuit(req: Annotated[PursuitRequest, Body()] = PursuitRequest()) -> dict:
    """Open the courtship gate and set the pursuit decoder's gains."""
    req = req or PursuitRequest()
    ctl = controller()
    with ctl.lock:
        p = ctl.motor.pursuit
        for field in ("arousal", "yaw_gain", "forward_gain", "stop_distance_m", "cruise_altitude_m", "enabled"):
            value = getattr(req, field)
            if value is not None:
                setattr(p, field, value)
        wired = bool(ctl.circuit.types("steer")) and bool(ctl.circuit.types("forward"))
        return {
            "pursuit": p.model_dump(),
            "decoder_wired": wired,
            "note": (
                "pursuit will steer"
                if wired
                else "outputs.steer and outputs.forward are still empty, so pursuit commands "
                "zero: pick descending types, set them with update_circuit, reload, calibrate."
            ),
        }


@router.post("/stop", operation_id="stop_drone")
def stop_drone() -> dict:
    """Zero the velocity, disarm the output and clear the escape clock. The panic button."""
    ctl = controller()
    cmd = ctl.stop()
    return {"command": cmd.model_dump(), "output_enabled": ctl.output_enabled}
