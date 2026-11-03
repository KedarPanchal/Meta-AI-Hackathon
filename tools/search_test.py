#!/usr/bin/env python3
"""
Drive the mission service (what the MCP tools call) over the live simulator, through the
ROS link, without an MCP client::

    python tools/search_test.py --look-only                # one detector call on the newest frame
    python tools/search_test.py --area -14 -50 18 -26      # a search mission over the person

Environment as for the MCP server: MISSION_LINK (ros), PERSON_DETECTOR, DETECT_COLOR, MUSE_*.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'mcp-server'))
os.environ.setdefault('MISSION_LINK', 'ros')

from app.bootstrap import ensure_fly_brain  # noqa: E402
ensure_fly_brain()
from mission.runner import MissionRequest, service  # noqa: E402
from mission.schemas import Area  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--area', type=float, nargs=4, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                    default=[-14.0, -50.0, 18.0, -26.0])
    ap.add_argument('--altitude', type=float, default=15.0)
    ap.add_argument('--mode', default='search')
    ap.add_argument('--timeout', type=float, default=900.0)
    ap.add_argument('--look-only', action='store_true')
    ap.add_argument('--no-confirm', action='store_true', help='declare FOUND on the first positive, no confirm pass')
    ap.add_argument('--speed', type=float, default=None, help='lane speed, m/s (default MISSION_SPEED_MPS)')
    ap.add_argument('--save-dir', default='.')
    args = ap.parse_args()

    svc = service()
    cfg = svc.cfg
    print('link:', svc.link.name, 'detector:', svc.detector.name, flush=True)
    t0 = time.time()
    while svc.link.pose() is None and time.time() - t0 < 60:
        time.sleep(0.5)
    print('pose:', svc.link.pose(), flush=True)
    t0 = time.time()
    while svc.link.frame() is None and time.time() - t0 < 30:
        time.sleep(0.5)
    look = svc.look()
    print('look:', json.dumps({k: v for k, v in look.items() if k != 'link'})[:500], flush=True)
    if args.look_only:
        return 0

    x0, y0, x1, y1 = args.area
    speed = args.speed or cfg.speed_mps
    req = MissionRequest(
        mode=args.mode, area=Area.box(x0, y0, x1, y1), altitude_m=args.altitude,
        speed_mps=speed, transit_speed_mps=max(cfg.transit_speed_mps, speed), lane_overlap=0.2,
        keepouts=svc.default_keepouts(), sweep_heading_deg=None, min_confidence=0.6,
        confirm=not args.no_confirm, return_home=False,
    )
    m = svc.start(req)
    print('plan:', json.dumps(m.plan.summary()), flush=True)
    seen_log = 0
    t0 = time.time()
    last_print = 0.0
    while time.time() - t0 < args.timeout:
        st = svc.status()
        for line in list(m._log)[seen_log:]:
            print('  log:', line, flush=True)
        seen_log = len(m._log)
        if time.time() - last_print > 10:
            last_print = time.time()
            b = st.brain
            print(f"  [{st.elapsed_s:.0f}s] state={st.state} wp={st.waypoint_index}/{st.n_waypoints} "
                  f"progress={st.progress:.2f} pos={st.position} brain={b.state if b else None} "
                  f"samples={st.samples} errors={st.detector_errors} frame_age={st.frame_age_s}", flush=True)
        if st.state in ('found', 'complete', 'failed', 'aborted'):
            break
        time.sleep(1.0)
    st = svc.status()
    print('FINAL:', st.state, st.note, flush=True)
    print('elapsed', st.elapsed_s, 'distance', st.distance_flown_m, 'samples', st.samples, flush=True)
    for s in st.sightings:
        print('  sighting:', s.estimated_xy, s.confirmed, s.n_sightings, s.detection.confidence,
              s.detection.description[:80], flush=True)
    fr = svc.link.frame()
    if fr is not None:
        import cv2
        path = os.path.join(args.save_dir, f'search_final_{st.state}.png')
        cv2.imwrite(path, fr.image)
        print('saved', path, flush=True)
    return 0 if st.state == 'found' else 1


if __name__ == '__main__':
    sys.exit(main())
