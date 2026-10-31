"""The five tools. Two start a flight, one watches it, one stops it, one looks once."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from mission.runner import MissionRequest, service
from mission.schemas import Area, Keepout, MissionStatus

router = APIRouter(prefix='/mission', tags=['mission'])


class SearchRequest(BaseModel):
    area: Area | None = Field(
        default=None,
        description='Box {x_min, x_max, y_min, y_max} or {polygon: [[x, y], ...]} in world metres. '
        'Default: the whole 120 x 120 m forest map.',
    )
    altitude_m: float | None = Field(
        default=None, ge=3.0, le=120.0,
        description='Flight level as world z. Default MISSION_ALTITUDE_M (15). The forest floor '
        'spans about -2.5..3 m and the tallest fixture tops out near 10.5 m. Lower flies a '
        'smaller camera footprint, so more lanes and a longer flight, but a bigger person.',
    )
    speed_mps: float | None = Field(default=None, gt=0.0, le=10.0, description='Lane speed, m/s.')
    lane_overlap: float | None = Field(
        default=None, ge=0.0, lt=0.9,
        description='Fraction of the camera footprint adjacent lanes share. Default 0.2.',
    )
    sweep_heading_deg: float | None = Field(
        default=None,
        description='Direction the lanes run, degrees from east. Default: along the longer side.',
    )
    keepouts: list[Keepout] | None = Field(
        default=None,
        description='Structures to route around, as circles {x, y, radius_m, top_z_m} or polygons. '
        'Default: the known towers and masts from the world file. Pass [] for none. Only those '
        'reaching the flight level are used.',
    )
    min_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Detector confidence that counts. Default 0.6.",
    )
    confirm_detections: bool | None = Field(
        default=None,
        description='After a positive, fly over the estimated spot and look straight down again '
        'before declaring the person found. Default on.',
    )
    dry_run: bool = Field(default=False, description='Plan only; nothing flies.')


class ScoutRequest(SearchRequest):
    return_home: bool = Field(
        default=False, description='Fly back to the start position after covering the area.'
    )


def _request(req: SearchRequest, mode: str) -> MissionRequest:
    svc = service()
    cfg = svc.cfg
    return MissionRequest(
        mode=mode,  # type: ignore[arg-type]
        area=req.area or cfg.default_area(),
        altitude_m=req.altitude_m if req.altitude_m is not None else cfg.altitude_m,
        speed_mps=req.speed_mps if req.speed_mps is not None else cfg.speed_mps,
        transit_speed_mps=max(cfg.transit_speed_mps, req.speed_mps or 0.0),
        lane_overlap=req.lane_overlap if req.lane_overlap is not None else cfg.lane_overlap,
        keepouts=req.keepouts if req.keepouts is not None else svc.default_keepouts(),
        sweep_heading_deg=req.sweep_heading_deg,
        min_confidence=req.min_confidence if req.min_confidence is not None else cfg.min_confidence,
        confirm=req.confirm_detections if req.confirm_detections is not None else cfg.confirm_detections,
        return_home=getattr(req, 'return_home', False),
    )


def _start(req: SearchRequest, mode: str) -> dict:
    svc = service()
    try:
        mreq = _request(req, mode)
        if req.dry_run:
            try:
                start = svc.link.pose()
            except RuntimeError:
                start = None
            plan = svc.plan(mreq, start)
            wps = [w.model_dump(exclude_none=True) for w in plan.waypoints]
            return {
                'dry_run': True,
                'mode': mode,
                'plan': plan.summary(),
                'cells': [c.model_dump() for c in plan.cells],
                'waypoints': wps if len(wps) <= 400 else wps[:400],
                'start_xy': start.xy() if start else None,
                'note': 'nothing was flown. Call again without dry_run to fly it.',
            }
        mission = svc.start(mreq)
    except (RuntimeError, ValueError) as exc:
        return {'error': str(exc), 'link': svc.link_name(), 'detector': svc.detector_name()}
    status = svc.status()
    return {
        'mission_id': mission.id,
        'mode': mode,
        'state': status.state,
        'link': svc.link.name,
        'detector': svc.detector.name,
        'plan': mission.plan.summary(),
        'first_waypoints': [w.model_dump(exclude_none=True) for w in mission.plan.waypoints[:6]],
        'note': (
            'flying. Poll mission_status; abort_mission stops it and holds position. '
            + ('The mission ends hovering over the person once a sighting is confirmed.'
               if mode == 'search' else 'Every sighting is kept; the mission ends after full coverage.')
        ),
    }


@router.post('/search', operation_id='search_for_person')
def search_for_person(req: Annotated[SearchRequest, Body()] = SearchRequest()) -> dict:
    """Find a missing person: lawnmower the area from above and stop over them when seen.

    Plans a boustrophedon coverage path for the area (lanes spaced by the downward camera's
    footprint at the flight level), hands the brain one waypoint at a time, and runs the Muse
    Spark vision model over the camera every few metres. A positive is confirmed by flying
    over the estimated spot and looking again; a confirmed sighting ends the mission hovering
    above the person, with their estimated world position in ``mission_status``.

    The brain keeps its reflexes throughout: an obstacle on the way produces an escape or a
    side-step and the drone then resumes closing on the waypoint. This tool never commands
    velocity.
    """
    return _start(req or SearchRequest(), 'search')


@router.post('/scout', operation_id='scout_area')
def scout_area(req: Annotated[ScoutRequest, Body()] = ScoutRequest()) -> dict:
    """Scout every part of the area: fly the full coverage path and report everyone seen.

    Same planner and camera pipeline as ``search_for_person``, but a sighting does not end the
    flight. Sightings closer than 6 m are merged. Ends holding position at the last waypoint, or
    back at the start with ``return_home``.
    """
    return _start(req or ScoutRequest(), 'scout')


@router.get('/status', operation_id='mission_status', response_model=MissionStatus)
def mission_status() -> MissionStatus:
    """Where the mission is: state, progress, position, the brain's goal, sightings and the log.

    ``state`` is one of idle, planned, running, found, complete, aborted, failed. ``best`` is the
    sighting to act on; ``estimated_xy`` is where the person is on the ground.
    """
    return service().status()


@router.post('/abort', operation_id='abort_mission')
def abort_mission() -> dict:
    """Stop the mission and hold position at the current altitude. Safe to call at any time."""
    svc = service()
    try:
        st = svc.abort()
    except RuntimeError as exc:
        return {'error': str(exc)}
    return {'state': st.state, 'position': st.position, 'brain': st.brain.model_dump() if st.brain else None,
            'note': 'holding position; a new search_for_person or scout_area may start now'}


@router.post('/look', operation_id='look_for_person')
def look_for_person() -> dict:
    """Run the person detector once on the newest camera frame, without moving.

    Use it to check what the camera sees right now, or to double-check a sighting after
    ``abort_mission``. Returns the detection, its estimated ground position, and the frame age.
    """
    svc = service()
    try:
        return svc.look()
    except RuntimeError as exc:
        return {'error': str(exc), 'link': svc.link_name(), 'detector': svc.detector_name()}
