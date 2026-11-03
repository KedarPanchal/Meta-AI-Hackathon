#!/usr/bin/env python3
"""
Flight tests against the live simulator, through the brain's own topics.

Run with the sim up (``ros2 launch fly_bringup brain_launch.py threats:=true threat_period_s:=0``)::

    python tools/flight_test.py rates                       # are the sensors and the brain alive
    python tools/flight_test.py reaction --launch '{"bearing_deg": 70}' --dump run.jsonl
    python tools/flight_test.py survey --x 2 --y -38 --altitudes 15 10 7

``reaction`` sends the drone on 40 m legs at the flight level and, once it is cruising, asks
the threat launcher for a sphere per ``--launch``; it reports when the brain escaped relative
to the sphere's time of flight (in simulation time), how close the sphere actually came (from
the launcher's reported positions against odometry) and whether the goal was reached after.
``survey`` hovers over a world point at several altitudes and saves both cameras' frames.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

from nav_msgs.msg import Odometry

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from sensor_msgs.msg import Image

from std_msgs.msg import String


class Harness(Node):
    def __init__(self, save_dir: str):
        super().__init__('flight_test')
        self.lock = threading.Lock()
        self.tel: list[dict] = []
        self.odom: list[tuple[float, float, float, float, float]] = []   # wall, x, y, z, yaw
        self.events: list[dict] = []
        self.img_count: dict[str, int] = {}
        self.img_last: dict[str, object] = {}
        self.save_dir = save_dir
        q = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(String, '/fly_brain/telemetry', self._tel, 10)
        self.create_subscription(Odometry, '/drone/odometry', self._odom, q)
        for t in ('/drone/camera_image', '/drone/camera_down_image'):
            self.create_subscription(Image, t, lambda m, t=t: self._img(m, t), q)
        self.create_subscription(String, '/threats/events', self._ev, 10)
        self.goal_pub = self.create_publisher(String, '/fly_brain/goal', 10)
        self.threat_pub = self.create_publisher(String, '/threats/launch', 10)

    def _tel(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        d['wall'] = time.time()
        with self.lock:
            self.tel.append(d)

    def _odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.odom.append((time.time(), p.x, p.y, p.z, yaw))

    def _img(self, msg, topic):
        with self.lock:
            self.img_count[topic] = self.img_count.get(topic, 0) + 1
            self.img_last[topic] = msg

    def _ev(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        d['wall'] = time.time()
        with self.lock:
            self.events.append(d)
        if d['event'] != 'pose':
            print(f"  [threat] {d['event']} {d['name']} ok={d['ok']}", flush=True)

    # --- helpers ---------------------------------------------------------------------
    def pose(self):
        with self.lock:
            return self.odom[-1] if self.odom else None

    def last_tel(self):
        with self.lock:
            return self.tel[-1] if self.tel else None

    def goto(self, x, y, z, label, speed=None):
        g = {'x': x, 'y': y, 'z': z, 'id': label, 'label': label}
        if speed:
            g['speed_mps'] = speed
        m = String()
        m.data = json.dumps(g)
        self.goal_pub.publish(m)
        print(f"  goal {label}: ({x:.1f}, {y:.1f}, {z:.1f})", flush=True)

    def wait_reached(self, timeout):
        t0 = time.time()
        # the brain's reached flag latches per goal: let the new goal clear it first
        while time.time() - t0 < 3.0:
            t = self.last_tel()
            if t and not t.get('goal_reached'):
                break
            time.sleep(0.1)
        while time.time() - t0 < timeout:
            t = self.last_tel()
            if t and t.get('goal_reached') and self.pose():
                return True
            time.sleep(0.2)
        return False

    def save_frame(self, topic, name):
        with self.lock:
            msg = self.img_last.get(topic)
        if msg is None:
            print(f"  no frame on {topic}", flush=True)
            return
        try:
            import cv2
            from cv_bridge import CvBridge
            img = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
            path = os.path.join(self.save_dir, name)
            cv2.imwrite(path, img)
            print(f"  saved {path} {img.shape}", flush=True)
        except Exception as exc:
            print(f"  save failed: {exc}", flush=True)


def wait_for(cond, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.2)
    print(f"  TIMEOUT waiting for {what}", flush=True)
    return False


def cmd_rates(h: Harness, args):
    print("waiting for odometry and telemetry...", flush=True)
    ok = wait_for(lambda: h.pose() is not None and h.last_tel() is not None, 90, 'odom+telemetry')
    with h.lock:
        n_tel, n_odom, imgs = len(h.tel), len(h.odom), dict(h.img_count)
    time.sleep(5.0)
    with h.lock:
        print(f"over 5 s: telemetry {(len(h.tel) - n_tel) / 5:.1f} Hz, odom {(len(h.odom) - n_odom) / 5:.1f} Hz, "
              + ', '.join(f"{k} {(v - imgs.get(k, 0)) / 5:.1f} Hz" for k, v in h.img_count.items()), flush=True)
    t = h.last_tel()
    print("pose:", h.pose() and [round(v, 2) for v in h.pose()[1:]], flush=True)
    print("telemetry:", json.dumps(t)[:400], flush=True)
    h.save_frame('/drone/camera_image', 'rates_front.png')
    h.save_frame('/drone/camera_down_image', 'rates_down.png')
    return 0 if ok else 1


def closest_approach(h: Harness, ev: dict, until: float):
    """Nearest the sphere came, from the launcher's own reported positions against odometry."""
    with h.lock:
        poses = [e for e in h.events if e['event'] == 'pose' and e['name'] == ev['name'] and e['wall'] <= until]
        od = [o for o in h.odom if ev['wall'] - 0.5 <= o[0] <= until]
    if not poses or not od:
        return (float('nan'), None)
    best = (1e9, None)
    j = 0
    for pe in poses:
        while j + 1 < len(od) and abs(od[j + 1][0] - pe['wall']) < abs(od[j][0] - pe['wall']):
            j += 1
        t, x, y, z, yaw = od[j]
        tx, ty, tz = pe['position']
        d = math.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
        if d < best[0]:
            best = (d, round(pe['wall'] - ev['wall'], 2))
    return best


def cmd_reaction(h: Harness, args):
    print("waiting for odometry and telemetry...", flush=True)
    if not wait_for(lambda: h.pose() is not None and h.last_tel() is not None, 90, 'odom+telemetry'):
        return 1
    p = h.pose()
    z = args.altitude
    ax, ay = p[1] + args.leg, p[2]
    h.goto(ax, ay, z, 'A')
    t0 = time.time()
    if not h.wait_reached(150):
        print("  goal A not reached", h.last_tel(), flush=True)
        return 1
    print(f"  reached A in {time.time() - t0:.1f} s; pose {[round(v, 1) for v in h.pose()[1:]]}", flush=True)
    results = []
    launches = [json.loads(s) for s in args.launch]
    for i, params in enumerate(launches):
        p = h.pose()
        bx = p[1] - args.leg if i % 2 == 0 else p[1] + args.leg
        label = f'B{i}'
        h.goto(bx, ay, z, label)

        def cruising():
            with h.lock:
                if len(h.odom) < 10:
                    return False
                a, b = h.odom[-10], h.odom[-1]
            dt = b[0] - a[0]
            return dt > 0 and math.hypot(b[1] - a[1], b[2] - a[2]) / dt > 1.5
        if not wait_for(cruising, 60, 'cruise'):
            return 1
        time.sleep(1.5)
        n_before = len(h.events)
        m = String()
        m.data = json.dumps(params)
        t_launch = time.time()
        h.threat_pub.publish(m)
        print(f"  launch {i}: {params}", flush=True)
        if not wait_for(lambda: any(e['event'] in ('spawned', 'spawn_failed') for e in h.events[n_before:]), 10, 'spawn event'):
            return 1
        ev = [e for e in h.events[n_before:] if e['event'] in ('spawned', 'spawn_failed')][0]
        if ev['event'] != 'spawned':
            print("  spawn failed", ev, flush=True)
            return 1
        tof = ev.get('tof_s') or ev['distance_m'] / ev['speed_mps']
        t_end = ev['wall'] + tof * 2.0 + 4.0
        first = {}
        while time.time() < t_end:
            with h.lock:
                recent = [t for t in h.tel if t['wall'] >= ev['wall']]
            for t in recent:
                if t['state'] in ('ESCAPE', 'AVOID') and t['state'] not in first:
                    first[t['state']] = t
            time.sleep(0.1)
        with h.lock:
            states = [t['state'] for t in h.tel if ev['wall'] <= t['wall'] <= t_end]
        seq = []
        for s in states:
            if not seq or seq[-1] != s:
                seq.append(s)
        d, at = closest_approach(h, ev, t_end)
        with h.lock:
            results_ev = [e for e in h.events if e['event'] == 'result' and e['name'] == ev['name']]
        verdict = results_ev[-1] if results_ev else None
        esc = first.get('ESCAPE')
        sim_latency = (esc['t'] - ev['sim_t']) if esc else None
        r = {
            'launch': params, 'spawned_at': round(ev['wall'] - t_launch, 2), 'tof_s': round(tof, 2),
            'escape_after_sim_s': None if sim_latency is None else round(sim_latency, 2),
            'to_contact_sim_s': None if sim_latency is None else round(tof - sim_latency, 2),
            'escape_reason': esc['reason'][:90] if esc else None,
            'escape_threat': esc['threat'] if esc else None,
            'escape_percept': esc['percept'] if esc else None,
            'closest_m': round(d, 2), 'closest_at_s': at, 'states': seq,
            'launcher_closest_m': verdict and verdict.get('closest_m'),
            'hit': verdict and verdict.get('hit'),
            'dodge': esc and esc.get('dodge'),
        }
        print("  RESULT", json.dumps(r), flush=True)
        results.append(r)
        if not h.wait_reached(120):
            print("  goal", label, "not reached after the threat", flush=True)
            r['resumed'] = False
        else:
            r['resumed'] = True
            print(f"  resumed and reached {label}", flush=True)
        time.sleep(2.0)
    print("SUMMARY")
    for r in results:
        print(json.dumps(r))
    return 0


def cmd_survey(h: Harness, args):
    print("waiting for odometry and telemetry...", flush=True)
    if not wait_for(lambda: h.pose() is not None and h.last_tel() is not None, 90, 'odom+telemetry'):
        return 1
    for z in args.altitudes:
        h.goto(args.x, args.y, z, f'survey_z{z:g}')
        if not h.wait_reached(180):
            print("  not reached", h.last_tel(), flush=True)
            return 1
        time.sleep(2.0)
        print(f"  at z={z:g}: pose {[round(v, 2) for v in h.pose()[1:]]}", flush=True)
        h.save_frame('/drone/camera_down_image', f'survey_down_z{z:g}.png')
        h.save_frame('/drone/camera_image', f'survey_front_z{z:g}.png')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['rates', 'reaction', 'survey'])
    ap.add_argument('--save-dir', default='.')
    ap.add_argument('--altitude', type=float, default=15.0)
    ap.add_argument('--leg', type=float, default=40.0)
    ap.add_argument('--launch', action='append', default=[], help='JSON params for one sphere; repeatable')
    ap.add_argument('--x', type=float, default=2.0)
    ap.add_argument('--y', type=float, default=-38.0)
    ap.add_argument('--altitudes', type=float, nargs='+', default=[15.0, 10.0, 7.0])
    ap.add_argument('--dump', default='', help='write every telemetry record, event and odometry sample here (jsonl)')
    args = ap.parse_args()
    rclpy.init()
    h = Harness(args.save_dir)
    threading.Thread(target=rclpy.spin, args=(h,), daemon=True).start()
    try:
        rc = {'rates': cmd_rates, 'reaction': cmd_reaction, 'survey': cmd_survey}[args.cmd](h, args)
    finally:
        if args.dump:
            with h.lock, open(args.dump, 'w') as f:
                for t in h.tel:
                    f.write(json.dumps({'kind': 'tel', **t}) + "\n")
                for e in h.events:
                    f.write(json.dumps({'kind': 'event', **e}) + "\n")
                for o in h.odom:
                    f.write(json.dumps({'kind': 'odom', 'wall': o[0], 'xyz': o[1:4], 'yaw': o[4]}) + "\n")
        h.destroy_node()
        rclpy.shutdown()
    sys.exit(rc)


if __name__ == '__main__':
    main()
