"""
The mission runner whose next move is chosen by Muse, not by a fixed lane order.

It keeps the coverage plan as a *candidate set*: a shortlist of real goals. Muse sees the
downward frame and the shortlist, then chooses ``goto`` (a candidate), ``investigate`` (a pixel
it believes holds the missing person), or ``hold``. The chosen pixel is projected to the ground
with the same camera model the detector uses, and the resulting goal carries an ``arousal``
value so the brain's pC1/DNp13 pathway is engaged. The brain still owns velocity, the escape
reflex and the safety clamp.

If Muse errors, the runner falls back to the next candidate and logs the failure; a bad API
answer can never leave the aircraft without a goal.
"""
from __future__ import annotations

import math
import time

from fly_brain.schemas import Goal

from .muse_navigator import NavDecision
from .runner import MissionRunner
from .schemas import Waypoint


class MuseMissionRunner(MissionRunner):
    """Muse chooses each goal; the coverage planner supplies candidate geometry."""

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
        m.navigator = 'muse'
        m.log(f'{m.mode} mission {m.id}: Muse navigator over {len(m.plan.waypoints)} candidate goals, '
              f'{m.plan.path_length_m:.0f} m at z={m.plan.altitude_m:.0f} m')
        remaining = list(m.plan.waypoints)
        last_sample_xy: tuple[float, float] | None = None
        idle = 0

        while remaining:
            if self._abort.is_set():
                break
            pose = link.pose()
            if pose is None:
                m.state = 'failed'
                m.note = 'lost odometry while Muse was navigating'
                m.log('FAILED: ' + m.note)
                return
            fr = link.frame()
            if fr is None:
                m.log('no downward frame yet; waiting before asking Muse')
                time.sleep(max(self.cfg.poll_s, 0.2))
                continue
            shortlist = remaining[:4]
            candidates = [
                {'id': self._candidate_id(w), 'x': round(w.x, 2), 'y': round(w.y, 2),
                 'z': round(w.z, 2), 'label': w.kind}
                for w in shortlist
            ]
            height = max(pose.z - self.cfg.ground_z_m, 1.0)
            decision = self.svc.muse_choose(fr.image, height, pose, candidates)
            m.nav_calls += 1
            m.last_nav_decision = decision.model_dump()
            if decision.error:
                m.log(f'Muse navigator error ({m.nav_errors}): {decision.error[:160]}')
                m.nav_errors += 1
                decision = NavDecision(action='goto', target_id=candidates[0]['id'],
                                       confidence=0.0, reason='fallback after Muse error')

            if decision.action == 'hold':
                idle += 1
                m.log(f'Muse holds: {decision.reason[:120]}')
                self._station_keep('Muse hold')
                if idle >= 3:
                    # Do not orbit forever on a hold; take the nearest candidate.
                    decision = NavDecision(action='goto', target_id=candidates[0]['id'],
                                           confidence=0.0, reason='progress fallback after repeated holds')
                else:
                    time.sleep(min(max(self.cfg.poll_s, 0.5), 2.0))
                    continue
            idle = 0

            if decision.action == 'investigate' and decision.target_pixel is not None:
                est = self.svc.camera.pixel_to_ground(
                    decision.target_pixel[0], decision.target_pixel[1],
                    pose.xyz(), pose.yaw, self.cfg.ground_z_m,
                )
                goal = Goal(
                    x=est[0], y=est[1], z=m.plan.altitude_m, speed_mps=self.cfg.speed_mps,
                    arrive_radius_m=1.0, id=f'{m.id}:muse-investigate-{m.nav_calls}',
                    label='Muse investigate',
                    arousal=max(0.25, decision.confidence),
                )
                m.log(f'Muse investigates pixel {decision.target_pixel} -> ({est[0]:.1f}, {est[1]:.1f}) '
                      f'conf {decision.confidence:.2f}: {decision.reason[:80]}')
                outcome = self._fly_goal(goal, 0, last_sample_xy)
                if (p := link.pose()) is not None:
                    last_sample_xy = p.xy()
                if outcome == 'found':
                    return
                # The pixel did not survive the look: drop the nearest candidate so the
                # next decision has new geometry rather than the same invitation.
                if shortlist:
                    remaining.remove(shortlist[0])
                continue

            chosen = next((w for w in shortlist if self._candidate_id(w) == decision.target_id), None)
            if chosen is None:
                chosen = shortlist[0] if shortlist else None
            if chosen is None:
                break
            goal = Goal(
                x=chosen.x, y=chosen.y, z=chosen.z, speed_mps=chosen.speed_mps,
                arrive_radius_m=chosen.arrive_radius_m,
                id=f'{m.id}:{self._candidate_id(chosen)}', label=f'Muse goto {chosen.kind}',
            )
            m.log(f'Muse goto {chosen.kind} {self._candidate_id(chosen)}: {decision.reason[:80]}')
            outcome = self._fly_goal(goal, 0, last_sample_xy)
            if (p := link.pose()) is not None:
                last_sample_xy = p.xy()
            if outcome == 'found':
                return
            if chosen in remaining:
                remaining.remove(chosen)

        if self._abort.is_set():
            m.state = 'aborted'
            m.note = 'aborted by operator'
            m.log('aborted; holding position')
            self._station_keep('hold after abort')
            return
        m.state = 'complete'
        m.waypoint_index = len(m.plan.waypoints)
        n = len(m.sightings)
        m.note = f'Muse finished the candidate set: {n} sighting(s)' if n else 'Muse finished the candidate set: nobody seen'
        m.log(m.note)
        if m.req.return_home and m.start_pose is not None:
            link.goto(Goal(x=m.start_pose.x, y=m.start_pose.y, z=m.plan.altitude_m,
                           id=f'{m.id}:home', label='home'))
            m.log('returning to the start position')
        else:
            self._station_keep('hold at end of Muse search')

    @staticmethod
    def _candidate_id(w: Waypoint) -> str:
        return f'{w.cell}:{w.lane}:{w.x:.1f}:{w.y:.1f}'

    def _fly_goal(self, goal: Goal, wp_index: int, last_sample_xy) -> str:
        """Fly one goal, sampling the camera along the way; return none/sighting/found/aborted."""
        m, link = self.m, self.svc.link
        link.goto(goal)
        t0 = time.time()
        resend_at = t0 + 1.0
        pose = link.pose()
        leg_len = (math.hypot(goal.x - pose.x, goal.y - pose.y) + abs(goal.z - pose.z)) if pose else 0.0
        speed = goal.speed_mps or self.cfg.speed_mps
        timeout = max(self.cfg.leg_timeout_min_s, leg_len / max(speed, 0.1) * self.cfg.leg_timeout_factor)
        while not self._abort.is_set():
            st = link.goal_status()
            p = link.pose()
            acked = st is not None and st.goal is not None and st.goal.id == goal.id
            if acked and st.reached:
                # one look directly above the chosen point
                if p is not None:
                    outcome = self._sample(wp_index, p)
                    if outcome == 'found':
                        return 'found'
                return 'reached'
            if p is not None and (last_sample_xy is None or
                                  math.hypot(p.x - last_sample_xy[0], p.y - last_sample_xy[1]) >= m.plan.sample_spacing_m):
                last_sample_xy = p.xy()
                outcome = self._sample(wp_index, p)
                if outcome == 'found':
                    return 'found'
            now = time.time()
            if not acked and now >= resend_at:
                link.goto(goal)
                resend_at = now + 2.0
            if now - t0 > timeout:
                m.timeouts += 1
                m.log(f'Muse goal {goal.label} not reached in {timeout:.0f} s; trying another candidate')
                return 'none'
            time.sleep(self.cfg.poll_s)
        return 'aborted'
