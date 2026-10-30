"""
Flying a plan: one goal to the brain at a time, the camera to the detector every few metres.

The runner is a thread. For each waypoint it publishes the goal, waits for the brain to report
it reached (re-sending if the brain has not acknowledged, because the first message to a
late-starting node is lost), and every ``sample_spacing_m`` of travel takes a frame and asks the
detector. A positive in **search** mode is confirmed by flying over the estimated position and
looking straight down again; a confirmed sighting ends the mission hovering over the person.
In **scout** mode positives are merged into sightings and the whole area is flown.

The mission never ends with the brain goal-less. Completion, abort and failure all end with a
station-keeping goal at the current position, so the drone holds its altitude instead of
falling back to the brain's low pursuit cruise height.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import threading
import time
import traceback
import uuid

from fly_brain.schemas import Goal

from .camera import DownCamera
from .config import MissionSettings, mission_settings
from .coverage import PlannerConfig, plan_coverage
from .detector import PersonDetector, make_detector
from .link import BrainLink, Pose, make_link
from .schemas import (
    Area,
    CoveragePlan,
    Detection,
    Keepout,
    MissionMode,
    MissionState,
    MissionStatus,
    Sighting,
    Waypoint,
)

SIGHTING_MERGE_M = 6.0
"""Positives closer than this are the same person."""


@dataclass
class MissionRequest:
    mode: MissionMode
    area: Area
    altitude_m: float
    speed_mps: float
    transit_speed_mps: float
    lane_overlap: float
    keepouts: list[Keepout]
    sweep_heading_deg: float | None
    min_confidence: float
    confirm: bool
    return_home: bool = False


class Mission:
    def __init__(self, req: MissionRequest, plan: CoveragePlan, start_pose: Pose | None):
        self.id = time.strftime('%H%M%S') + '-' + uuid.uuid4().hex[:4]
        self.req = req
        self.mode: MissionMode = req.mode
        self.plan = plan
        self.start_pose = start_pose
        self.state: MissionState = 'planned'
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.waypoint_index = 0
        self.distance_flown_m = 0.0
        self.samples = 0
        self.stale_frames = 0
        self.detector_errors = 0
        self.last_detector_error: str | None = None
        self.timeouts = 0
        self.false_alarms = 0
        self.sightings: list[Sighting] = []
        self.note = ''
        self._log: deque[str] = deque(maxlen=80)
        self.lock = threading.RLock()
        self._cum = [0.0]
        prev = (start_pose.x, start_pose.y) if start_pose else (plan.waypoints[0].x, plan.waypoints[0].y) if plan.waypoints else (0.0, 0.0)
        for w in plan.waypoints:
            self._cum.append(self._cum[-1] + math.hypot(w.x - prev[0], w.y - prev[1]))
            prev = (w.x, w.y)

    def log(self, msg: str) -> None:
        with self.lock:
            self._log.append(f'{time.strftime("%H:%M:%S")} {msg}')

    def progress(self, horizontal_to_goal: float | None) -> float:
        total = self._cum[-1]
        if total <= 0:
            return 1.0 if self.state in ('complete', 'found') else 0.0
        if self.state in ('complete', 'found'):
            return 1.0
        i = min(self.waypoint_index, len(self.plan.waypoints) - 1)
        done = self._cum[i]
        leg = self._cum[i + 1] - self._cum[i] if i + 1 < len(self._cum) else 0.0
        if horizontal_to_goal is not None and leg > 0:
            done += max(0.0, min(leg, leg - horizontal_to_goal))
        return max(0.0, min(1.0, done / total))

    @property
    def best(self) -> Sighting | None:
        if not self.sightings:
            return None
        return max(self.sightings, key=lambda s: (s.confirmed, s.detection.confidence, s.n_sightings))


class MissionRunner(threading.Thread):
    def __init__(self, svc: 'MissionService', mission: Mission):
        super().__init__(name=f'mission-{mission.id}', daemon=True)
        self.svc = svc
        self.m = mission
        self.cfg = svc.cfg
        self._abort = threading.Event()

    def abort(self) -> None:
        self._abort.set()

    # --- the flight -------------------------------------------------------------------

    def run(self) -> None:
        m = self.m
        try:
            self._fly()
        except Exception as exc:
            m.state = 'failed'
            m.note = f'{type(exc).__name__}: {exc}'
            m.log('FAILED: ' + m.note)
            m.log(traceback.format_exc().splitlines()[-1])
            self._station_keep('hold after failure')
        finally:
            m.finished_at = m.finished_at or time.time()

    def _fly(self) -> None:
        m, link = self.m, self.svc.link
        pose = self._wait_pose(10.0)
        if pose is None:
            m.state = 'failed'
            m.note = (f'no odometry from the brain link ({link.name}): is fly_brain_pilot running '
                      f'and publishing {self.cfg.brain.topic_odom}?')
            m.log('FAILED: ' + m.note)
            return
        m.started_at = time.time()
        m.state = 'running'
        m.log(f'{m.mode} mission {m.id}: {len(m.plan.waypoints)} waypoints, '
              f'{m.plan.path_length_m:.0f} m, {m.plan.n_lanes} lanes at z={m.plan.altitude_m:.0f} m')
        last_pose = pose
        last_sample_xy: tuple[float, float] | None = None
        consecutive_timeouts = 0

        for i, wp in enumerate(m.plan.waypoints):
            if self._abort.is_set():
                break
            m.waypoint_index = i
            goal = self._goal_for(i, wp)
            link.goto(goal)
            t0 = time.time()
            leg_len = math.hypot(wp.x - last_pose.x, wp.y - last_pose.y) + abs(wp.z - last_pose.z)
            speed = wp.speed_mps or self.cfg.speed_mps
            timeout = max(self.cfg.leg_timeout_min_s, leg_len / max(speed, 0.1) * self.cfg.leg_timeout_factor)
            resend_at = t0 + 1.0
            reached = False
            while not self._abort.is_set():
                st = link.goal_status()
                p = link.pose()
                if p is not None:
                    m.distance_flown_m += math.hypot(p.x - last_pose.x, p.y - last_pose.y)
                    last_pose = p
                acked = st is not None and st.goal is not None and st.goal.id == goal.id
                if acked and st.reached:
                    reached = True
                    break
                now = time.time()
                if not acked and now >= resend_at:
                    link.goto(goal)
                    resend_at = now + 2.0
                if now - t0 > timeout:
                    break
                if p is not None and (last_sample_xy is None
                                      or math.hypot(p.x - last_sample_xy[0], p.y - last_sample_xy[1]) >= m.plan.sample_spacing_m):
                    last_sample_xy = p.xy()
                    outcome = self._sample(i, p)
                    if outcome == 'found':
                        return
                    if outcome == 'diverted':          # confirm flight happened; resume this leg
                        link.goto(goal)
                        t0 = time.time()
                        resend_at = t0 + 1.0
                time.sleep(self.cfg.poll_s)
            if self._abort.is_set():
                break
            if reached:
                consecutive_timeouts = 0
            else:
                m.timeouts += 1
                consecutive_timeouts += 1
                m.log(f'waypoint {i} ({wp.kind}) not reached in {timeout:.0f} s; moving on')
                if consecutive_timeouts >= 3:
                    m.state = 'failed'
                    m.note = ('three waypoints in a row timed out: the brain is not flying the goals. '
                              'Check it is armed (arm_on_goal) and that odometry is live.')
                    m.log('FAILED: ' + m.note)
                    self._station_keep('hold after failure')
                    return

        if self._abort.is_set():
            m.state = 'aborted'
            m.note = 'aborted by operator'
            m.log('aborted; holding position')
            self._station_keep('hold after abort')
            return
        m.state = 'complete'
        m.waypoint_index = len(m.plan.waypoints)
        n = len(m.sightings)
        m.note = (f'covered the area: {n} sighting(s)' if n else 'covered the area: nobody seen')
        m.log(m.note)
        if m.req.return_home and m.start_pose is not None:
            home = Goal(x=m.start_pose.x, y=m.start_pose.y, z=m.plan.altitude_m, id=f'{m.id}:home', label='home')
            link.goto(home)
            m.log('returning to the start position')
        else:
            self._station_keep('hold at end of coverage')

    # --- helpers -------------------------------------------------------------------------

    def _goal_for(self, i: int, wp: Waypoint) -> Goal:
        n = len(self.m.plan.waypoints)
        return Goal(
            x=wp.x, y=wp.y, z=wp.z, speed_mps=wp.speed_mps, arrive_radius_m=wp.arrive_radius_m,
            id=f'{self.m.id}:{i}', label=f'{wp.kind} {i + 1}/{n}',
        )

    def _wait_pose(self, timeout_s: float) -> Pose | None:
        t0 = time.time()
        while time.time() - t0 < timeout_s and not self._abort.is_set():
            p = self.svc.link.pose()
            if p is not None:
                return p
            time.sleep(0.1)
        return self.svc.link.pose()

    def _station_keep(self, label: str) -> None:
        p = self.svc.link.pose()
        if p is None:
            return
        z = self.m.plan.altitude_m if self.m.started_at else p.z
        self.svc.link.goto(Goal(x=p.x, y=p.y, z=z, id=f'{self.m.id}:hold', label=label))

    def _wait_reached(self, goal: Goal, timeout_s: float) -> bool:
        t0 = time.time()
        resend_at = t0 + 1.0
        while time.time() - t0 < timeout_s and not self._abort.is_set():
            st = self.svc.link.goal_status()
            acked = st is not None and st.goal is not None and st.goal.id == goal.id
            if acked and st.reached:
                return True
            if not acked and time.time() >= resend_at:
                self.svc.link.goto(goal)
                resend_at = time.time() + 2.0
            time.sleep(self.cfg.poll_s)
        return False

    def _sample(self, wp_index: int, pose: Pose) -> str:
        """Look once. Returns 'none', 'sighting', 'diverted' (confirm flown, not found) or 'found'."""
        m = self.m
        det = self.svc.detect_now(pose)
        if det is None:
            m.stale_frames += 1
            return 'none'
        m.samples += 1
        if det.error:
            m.detector_errors += 1
            m.last_detector_error = det.error
            m.log(f'detector error ({m.detector_errors}): {det.error[:120]}')
            return 'none'
        if not (det.person_visible and det.confidence >= m.req.min_confidence):
            return 'none'
        est = self.svc.locate(det, pose)
        s = Sighting(t=time.time(), detection=det, drone_position=pose.xyz(), drone_yaw_rad=pose.yaw,
                     estimated_xy=est, waypoint_index=wp_index)
        self._record(s)
        where = f'({est[0]:.1f}, {est[1]:.1f})' if est else 'under the drone'
        m.log(f'PERSON? conf {det.confidence:.2f} near {where}: {det.description[:80]}')
        if m.mode != 'search':
            return 'sighting'
        if not m.req.confirm:
            return self._found(s, est or pose.xy())
        return self._confirm(s, est or pose.xy())

    def _confirm(self, s: Sighting, est: tuple[float, float]) -> str:
        m, link = self.m, self.svc.link
        k = m.false_alarms + 1
        goal = Goal(x=est[0], y=est[1], z=m.plan.altitude_m, speed_mps=self.cfg.speed_mps,
                    arrive_radius_m=1.0, id=f'{m.id}:confirm{k}', label=f'confirm sighting {k}')
        m.log(f'confirming: flying over ({est[0]:.1f}, {est[1]:.1f})')
        link.goto(goal)
        reached = self._wait_reached(goal, max(self.cfg.leg_timeout_min_s, 60.0))
        if self._abort.is_set():
            return 'diverted'
        if not reached:
            m.log('confirm flight timed out; keeping the sighting unconfirmed')
            return 'diverted'
        positives = 0
        best: Detection | None = None
        for _ in range(3):
            p = link.pose()
            det = self.svc.detect_now(p) if p else None
            if det is not None:
                m.samples += 1
                if det.error:
                    m.detector_errors += 1
                    m.last_detector_error = det.error
                elif det.person_visible and det.confidence >= m.req.min_confidence:
                    positives += 1
                    if best is None or det.confidence > best.confidence:
                        best = det
                        if p is not None:
                            s.estimated_xy = self.svc.locate(det, p) or s.estimated_xy
            time.sleep(0.3)
        if positives and best is not None:
            s.detection = best
            s.confirmed = True
            s.n_sightings += positives
            return self._found(s, s.estimated_xy or est)
        m.false_alarms += 1
        m.log(f'not confirmed from above ({positives}/3 positive); resuming the search')
        return 'diverted'

    def _found(self, s: Sighting, est: tuple[float, float]) -> str:
        m, link = self.m, self.svc.link
        m.state = 'found'
        m.finished_at = time.time()
        m.note = (f'person found near ({est[0]:.1f}, {est[1]:.1f}) with confidence '
                  f'{s.detection.confidence:.2f}' + (' (confirmed from above)' if s.confirmed else ''))
        m.log('FOUND: ' + m.note)
        link.goto(Goal(x=est[0], y=est[1], z=m.plan.altitude_m, id=f'{m.id}:found', label='over person'))
        return 'found'

    def _record(self, s: Sighting) -> None:
        m = self.m
        with m.lock:
            for other in m.sightings:
                if s.estimated_xy and other.estimated_xy and \
                        math.hypot(s.estimated_xy[0] - other.estimated_xy[0],
                                   s.estimated_xy[1] - other.estimated_xy[1]) <= SIGHTING_MERGE_M:
                    other.n_sightings += 1
                    if s.detection.confidence > other.detection.confidence:
                        other.detection = s.detection
                        other.estimated_xy = s.estimated_xy
                        other.drone_position = s.drone_position
                        other.drone_yaw_rad = s.drone_yaw_rad
                        other.t = s.t
                    return
            m.sightings.append(s)


class MissionService:
    """One link, one detector, at most one mission in flight. The MCP tools talk to this."""

    def __init__(self, cfg: MissionSettings | None = None):
        self.cfg = cfg or mission_settings()
        self.camera = DownCamera(self.cfg.cam_hfov_deg, self.cfg.cam_width, self.cfg.cam_height)
        self._link: BrainLink | None = None
        self._detector: PersonDetector | None = None
        self.link_error: str | None = None
        self.detector_error: str | None = None
        self.mission: Mission | None = None
        self.runner: MissionRunner | None = None
        self.lock = threading.RLock()

    # --- lazy parts ------------------------------------------------------------------

    @property
    def link(self) -> BrainLink:
        with self.lock:
            if self._link is None:
                try:
                    self._link = make_link(self.cfg)
                    self.link_error = None
                except Exception as exc:
                    self.link_error = f'{type(exc).__name__}: {exc}'
                    raise RuntimeError(f'cannot reach the brain ({self.cfg.link}): {self.link_error}') from exc
            return self._link

    @property
    def detector(self) -> PersonDetector:
        with self.lock:
            if self._detector is None:
                try:
                    self._detector = make_detector(self.cfg)
                    self.detector_error = None
                except Exception as exc:
                    self.detector_error = f'{type(exc).__name__}: {exc}'
                    raise RuntimeError(f'no person detector: {self.detector_error}') from exc
            return self._detector

    def link_name(self) -> str:
        if self._link is not None:
            return self._link.name
        return f'not connected ({self.link_error})' if self.link_error else f'{self.cfg.link} (not connected yet)'

    def detector_name(self) -> str:
        if self._detector is not None:
            return self._detector.name
        if self.detector_error:
            return f'unavailable ({self.detector_error})'
        kind = self.cfg.detector
        if kind == 'auto':
            kind = 'muse' if self.cfg.muse_api_key else 'color'
        return f'{kind} (not loaded yet)'

    # --- planning -------------------------------------------------------------------------

    def default_keepouts(self) -> list[Keepout]:
        path = self.cfg.keepouts_file
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding='utf-8'))
        items = data.get('keepouts', data) if isinstance(data, dict) else data
        return [Keepout.model_validate(k) for k in items]

    def plan(self, req: MissionRequest, start: Pose | None) -> CoveragePlan:
        used: list[list[tuple[float, float]]] = []
        ignored: list[str] = []
        for k in req.keepouts:
            if k.blocks_level(req.altitude_m, self.cfg.keepout_vertical_clearance_m):
                used.append(k.footprint(self.cfg.keepout_clearance_m))
            else:
                ignored.append(k.name or 'unnamed')
        if start is not None:
            start_xy, start_z = start.xy(), start.z
        else:
            start_xy, start_z = (self.cfg.sim_start[0], self.cfg.sim_start[1]), self.cfg.sim_start[2]
        pc = PlannerConfig(
            altitude_m=req.altitude_m, ground_z_m=self.cfg.ground_z_m, camera=self.camera,
            lane_overlap=req.lane_overlap, sweep_heading_deg=req.sweep_heading_deg,
            lane_speed_mps=req.speed_mps, transit_speed_mps=req.transit_speed_mps,
            lane_arrive_radius_m=self.cfg.arrive_radius_m,
            transit_arrive_radius_m=self.cfg.transit_arrive_radius_m,
            sample_fraction=self.cfg.sample_fraction, start_z_m=start_z,
        )
        return plan_coverage(req.area.vertices(), used, start_xy, pc, keepouts_ignored=ignored)

    # --- missions -----------------------------------------------------------------------

    def start(self, req: MissionRequest) -> Mission:
        with self.lock:
            if self.mission is not None and self.mission.state in ('planned', 'running'):
                raise RuntimeError(
                    f'mission {self.mission.id} is still {self.mission.state}; abort_mission first'
                )
            link = self.link
            _ = self.detector                              # fail now, not mid-flight
            start = link.pose()
            plan = self.plan(req, start)
            if not plan.waypoints:
                raise RuntimeError('the plan has no waypoints: ' + '; '.join(plan.warnings))
            mission = Mission(req, plan, start)
            self.mission = mission
            self.runner = MissionRunner(self, mission)
            self.runner.start()
            return mission

    def abort(self) -> MissionStatus:
        with self.lock:
            runner, mission = self.runner, self.mission
        parked = False
        if runner is not None and runner.is_alive():
            runner.abort()
            runner.join(timeout=5.0)
            parked = not runner.is_alive()          # the runner ends with a station-keep goal
        elif mission is not None and mission.state in ('planned', 'running'):
            mission.state = 'aborted'
        if not parked:
            # nothing was in flight: still park the drone where it is, which is what "abort" means
            try:
                p = self.link.pose()
                if p is not None:
                    self.link.goto(Goal(x=p.x, y=p.y, z=p.z, id='hold', label='hold (abort with no mission)'))
            except RuntimeError:
                pass
        return self.status()

    def status(self) -> MissionStatus:
        with self.lock:
            m = self.mission
        st = MissionStatus(link=self.link_name(), detector=self.detector_name())
        pose = brain = frame = None
        if self._link is not None:
            pose = self._link.pose()
            brain = self._link.goal_status()
            frame = self._link.frame() if self._link.name != 'sim' else None
            if frame is not None:
                st.frame_age_s = round(time.time() - frame.t, 2)
            elif self._link.name == 'sim':
                st.frame_age_s = 0.0
        if pose is not None:
            st.position = (round(pose.x, 2), round(pose.y, 2), round(pose.z, 2))
            st.yaw_rad = round(pose.yaw, 3)
        st.brain = brain
        if m is None:
            st.note = 'no mission yet. search_for_person or scout_area starts one.'
            return st
        with m.lock:
            st.id, st.mode, st.state = m.id, m.mode, m.state
            st.started_at, st.finished_at = m.started_at, m.finished_at
            if m.started_at:
                st.elapsed_s = round((m.finished_at or time.time()) - m.started_at, 1)
            st.waypoint_index = m.waypoint_index
            st.n_waypoints = len(m.plan.waypoints)
            st.progress = round(m.progress(brain.horizontal_m if brain and brain.goal else None), 3)
            st.distance_flown_m = round(m.distance_flown_m, 1)
            st.samples, st.detector_errors = m.samples, m.detector_errors
            st.last_detector_error = m.last_detector_error
            st.sightings = list(m.sightings)
            st.best = m.best
            st.plan = m.plan.summary()
            st.next_waypoints = m.plan.waypoints[m.waypoint_index: m.waypoint_index + 4]
            st.log = list(m._log)[-15:]
            st.note = m.note
        return st

    # --- looking ------------------------------------------------------------------------------

    def detect_now(self, pose: Pose | None) -> Detection | None:
        """One detector call on the newest frame, or None if there is no fresh frame."""
        fr = self.link.frame()
        if fr is None or time.time() - fr.t > self.cfg.max_frame_age_s:
            return None
        height = (pose.z if pose else self.cfg.altitude_m) - self.cfg.ground_z_m
        return self.detector.detect(fr.image, max(height, 1.0))

    def locate(self, det: Detection, pose: Pose) -> tuple[float, float] | None:
        c = det.centre()
        if c is None:
            return (round(pose.x, 2), round(pose.y, 2))
        x, y = self.camera.pixel_to_ground(c[0], c[1], pose.xyz(), pose.yaw, self.cfg.ground_z_m)
        return (round(x, 2), round(y, 2))

    def look(self) -> dict:
        link = self.link
        pose = link.pose()
        fr = link.frame()
        if fr is None:
            return {'error': f'no camera frame yet from the {link.name} link', 'link': link.describe()}
        age = time.time() - fr.t
        height = (pose.z if pose else self.cfg.altitude_m) - self.cfg.ground_z_m
        det = self.detector.detect(fr.image, max(height, 1.0))
        out = {
            'detection': det.model_dump(),
            'frame_age_s': round(age, 2),
            'frame_shape': list(getattr(fr.image, 'shape', [])),
            'drone_position': pose.xyz() if pose else None,
            'drone_yaw_rad': pose.yaw if pose else None,
            'estimated_xy': self.locate(det, pose) if (pose and det.person_visible and not det.error) else None,
            'detector': self.detector.name,
            'link': link.name,
        }
        if det.error:
            out['note'] = 'the detector failed; nothing above the error line is meaningful'
        return out

    def close(self) -> None:
        with self.lock:
            if self.runner is not None and self.runner.is_alive():
                self.runner.abort()
                self.runner.join(timeout=5.0)
            if self._link is not None:
                self._link.close()
                self._link = None


_service: MissionService | None = None
_service_lock = threading.Lock()


def service() -> MissionService:
    global _service
    with _service_lock:
        if _service is None:
            _service = MissionService()
        return _service


def reset_service() -> None:
    """Drop the singleton, stopping any flight and link. For tests."""
    global _service
    with _service_lock:
        if _service is not None:
            _service.close()
        _service = None
