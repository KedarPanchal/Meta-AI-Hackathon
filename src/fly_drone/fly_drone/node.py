"""
The rclpy node: sensors in, one brain tick per control period, velocity out.

Shape of a cycle::

    Image  --detect---> boxes ----+
    Lidar  --cluster--> threats --+--LoomTracker--> Percept --encode--> spikes --motor--> Twist
    Odometry -------------------------------------^ (altitude, heading)           ^
    Audio   --------------------------------------^ (optional; synthesised)       |
    Goal (JSON on /fly_brain/goal) -----------------------------------------------'

Two sensors feed the loom reflex and the brain cannot tell them apart. The forward camera
gives colour or motion blobs in pixels (:mod:`fly_drone.detect`); the single-plane lidar on the
airframe gives range clusters all the way round (:mod:`fly_drone.lidar`). Both become
:class:`fly_brain.perception.Threat` objects -- bearing, elevation, angular size -- and the
tracker differentiates angular size per object to get the expansion rate the giant fibre
responds to. The lidar is the honest one about range, so a lidar threat carries ``distance_m``
and the escape log gets a true time-to-contact.

The goal is the mission layer's only input: a world position, as a ``fly_brain.schemas.Goal``
serialised to JSON on a ``std_msgs/String``. The node answers with a ``GoalStatus`` on
``/fly_brain/goal_status`` every tick. Receiving a goal arms the output (``arm_on_goal``), because
a goal is an explicit instruction to fly. A one-line JSON telemetry record goes out on
``/fly_brain/telemetry`` every tick: the percept, the spikes, the command and its state, which
is what a test watches for ESCAPE.

**Output is off until something arms it.** The node starts with the controller's output gate
closed, so it can be run against a live simulator to watch what the brain would do before
anything is allowed to move. Arm it with the ``enable_output`` launch argument or ROS
parameter, or send a goal.

Why the brain does not run in the subscription callback: a tick costs tens of milliseconds
(see the measured ``run()`` overhead in ``REFLEX.md``), which would block the executor and
back up the image queue. It runs on a timer instead, on the latest frame only, and old frames
are dropped rather than queued.

Time: everything perceptual is stamped with the node's clock, so under ``use_sim_time`` the
expansion rates, the escape's hold and cooldown and the control period all follow Gazebo's
clock however slowly it renders.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time

from fly_brain.config import reload_settings
from fly_brain.controller import controller
from fly_brain.perception import Threat, threat_from_box
from fly_brain.schemas import Box, Goal, Percept

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSPresetProfiles

from .bridge import make_bridge
from .detect import Detector
from .lidar import LidarThreatDetector, pointcloud_xyz

try:
    from .dodge_advisor import MuseDodgeAdvisor
except ImportError:                     # httpx absent
    MuseDodgeAdvisor = None

try:                                    # not in this workspace's environment; optional
    from audio_common_msgs.msg import AudioData
    HAVE_AUDIO = True
except ImportError:
    AudioData = None
    HAVE_AUDIO = False

LIDAR_MAX_AGE_S = 0.5
"""A lidar scan older than this is not evidence of anything any more."""
FRAME_MAX_AGE_S = 1.0
"""Likewise a camera frame."""


class BrainDroneNode(Node):
    """Wires the shared :class:`fly_brain.Controller` to ROS topics."""

    def __init__(self) -> None:
        super().__init__('fly_brain_pilot')
        self.cfg = reload_settings()
        self.ctl = controller()
        self.ctl.cfg = self.cfg

        self.declare_parameter('enable_output', False)
        self.declare_parameter('arm_on_goal', True)
        self.declare_parameter('arousal', 0.0)
        self.declare_parameter('audio_mode', 'synthetic' if not HAVE_AUDIO else 'topic')

        self._lock = threading.Lock()
        self._frame = None               # latest image, BGR
        self._frame_t = 0.0
        self._sound_db = -80.0
        self.detector = Detector()
        self.lidar = LidarThreatDetector(
            max_range_m=self.cfg.lidar_max_range_m, max_below_m=self.cfg.lidar_max_below_m,
            max_above_m=self.cfg.lidar_max_above_m,
        )
        self._lidar_threats: list[Threat] = []
        self._lidar_t = 0.0
        self._lidar_scans = 0
        self._lidar_points = 0
        self._bridge_cv = None
        self._hfov_logged = False
        self.ctl.tracker.hfov_deg = self.cfg.camera_hfov_deg

        # The connectome runtime is pure NumPy and has no import-time signal handler,
        # so it is safe to build lazily from the control timer. This log line points at
        # the vendored data so an operator can see there is no sibling checkout.
        self.get_logger().info(
            f'brain backend={self.cfg.backend} data={self.cfg.connectome_file()}'
        )

        self.bridge = make_bridge(
            self, self.cfg.ros_bridge, self.cfg.topic_cmd_vel, self.cfg.frame_id,
            enable_topic=self.cfg.topic_enable,
        )
        self.ctl.set_bridge(self.bridge)
        self.ctl.enable_output(bool(self.get_parameter('enable_output').value))

        # The bridge publishes RELIABLE, and a best-effort subscription to a large
        # Image or PointCloud2 loses essentially every message on the macOS loopback
        # (measured 0/51 frames best-effort vs 51/51 reliable in 20 s): without a
        # retransmit, one dropped fragment drops the frame and the reflexes go blind.
        # Images and lidar subscribe reliably; odometry, camera info and audio are
        # small and high-rate, so they stay best-effort on purpose.
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2
        from std_msgs.msg import String

        self._String = String
        self.create_subscription(Odometry, self.cfg.topic_odom, self._on_odom, sensor_qos)
        self.create_subscription(Image, self.cfg.topic_image, self._on_image, 10)
        self.create_subscription(
            CameraInfo, self.cfg.topic_camera_info, self._on_camera_info, sensor_qos
        )
        if self.cfg.lidar_enabled:
            self.create_subscription(
                PointCloud2, self.cfg.topic_lidar, self._on_lidar, 10
            )
        self.create_subscription(String, self.cfg.topic_goal, self._on_goal, 10)
        self.status_pub = self.create_publisher(String, self.cfg.topic_goal_status, 10)
        self.telemetry_pub = self.create_publisher(String, self.cfg.topic_telemetry, 10)
        self.muse_raw_pub = self.create_publisher(String, self.cfg.topic_muse_raw, 10)
        # Muse as the dodge advisor, when there is a key: asked about each moving object
        # ahead of time, its answer decides the escape's direction if it arrives in time.
        # Its complete answer also goes out on topic_muse_raw for debugging.
        self.advisor = None
        self._last_advice = None
        want = os.environ.get('FLYDRONE_DODGE_ADVISOR', '1').strip().lower() in ('1', 'true', 'yes', 'on')
        if want and MuseDodgeAdvisor is not None and (os.environ.get('MUSE_API_KEY') or os.environ.get('MODEL_API_KEY')):
            try:
                self.advisor = MuseDodgeAdvisor(on_raw=self._publish_muse_raw)
                self.get_logger().info(f'Muse dodge advisor on ({self.advisor.model})')
            except Exception as exc:
                self.get_logger().warning(f'no Muse dodge advisor: {exc}')
        if HAVE_AUDIO and self.get_parameter('audio_mode').value == 'topic':
            self.create_subscription(AudioData, self.cfg.topic_audio, self._on_audio, sensor_qos)

        period = 1.0 / max(self.cfg.loop_hz, 0.1)
        self.timer = self.create_timer(period, self._on_tick)
        self.get_logger().info(
            f'fly_brain_pilot up: backend={self.cfg.backend} bridge={self.bridge.name} '
            f'cmd={self.cfg.topic_cmd_vel} odom={self.cfg.topic_odom} '
            f'image={self.cfg.topic_image} '
            f"lidar={self.cfg.topic_lidar if self.cfg.lidar_enabled else 'off'} "
            f'goal={self.cfg.topic_goal} loop={self.cfg.loop_hz} Hz '
            f'tick={self.cfg.brain_tick_ms} ms '
            f"output={'ARMED' if self.ctl.output_enabled else 'disabled'}"
        )

    # --- subscriptions -------------------------------------------------------

    def _on_odom(self, msg) -> None:
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        q = msg.pose.pose.orientation
        st = self.ctl.state
        st.connected = True
        st.position = (p.x, p.y, p.z)
        st.velocity = (v.x, v.y, v.z)
        st.altitude_m = p.z
        st.yaw_rad = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        # roll and pitch matter to the lidar: its scan plane tilts with the airframe
        st.roll_rad = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x**2 + q.y**2))
        st.pitch_rad = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))

    def _on_image(self, msg) -> None:
        """Keep only the newest frame: the brain cannot keep up with the camera."""
        img = self._to_cv(msg)
        if img is None:
            return
        with self._lock:
            self._frame = img
            self._frame_t = self._now()

    def _on_camera_info(self, msg) -> None:
        """The real horizontal field of view, from the intrinsics: HFOV = 2 atan(w / 2 fx)."""
        k = list(msg.k) if hasattr(msg, 'k') else []
        fx = float(k[0]) if len(k) >= 5 else 0.0
        if fx <= 0.0 or int(msg.width) <= 0:
            return
        hfov = math.degrees(2.0 * math.atan(float(msg.width) / (2.0 * fx)))
        if abs(hfov - self.ctl.tracker.hfov_deg) > 0.05:
            self.ctl.tracker.hfov_deg = hfov
            self.cfg.camera_hfov_deg = hfov
        if not self._hfov_logged:
            self._hfov_logged = True
            self.get_logger().info(
                f'camera_info: {msg.width}x{msg.height}, fx {fx:.1f} -> HFOV {hfov:.1f} deg'
            )

    def _on_lidar(self, msg) -> None:
        """One scan to clusters to threats. Needs the heading, so waits for odometry."""
        st = self.ctl.state
        if not st.connected:
            return
        pts = pointcloud_xyz(msg)
        t = self._now()
        threats = self.lidar.update(
            pts, (st.position[0], st.position[1]), st.yaw_rad, t,
            roll_rad=st.roll_rad, pitch_rad=st.pitch_rad, drone_z=st.position[2],
        )
        with self._lock:
            self._lidar_threats = threats
            self._lidar_t = t
            self._lidar_scans += 1
            self._lidar_points = int(len(pts))

    def _on_goal(self, msg) -> None:
        """A goal from the mission layer. Parsed strictly: a malformed goal is refused, not fixed.

        Receiving one arms the output when ``arm_on_goal`` is set: a goal is an order to fly.
        """
        try:
            goal = Goal.model_validate_json(msg.data)
        except Exception as exc:
            self.get_logger().error(f'bad goal on {self.cfg.topic_goal}: {exc}')
            return
        self.ctl.set_goal(goal)
        if bool(self.get_parameter('arm_on_goal').value) and not self.ctl.output_enabled:
            # the parameter is the source of truth each tick, so flip it rather than the gate
            self.set_parameters([Parameter('enable_output', Parameter.Type.BOOL, True)])
            self.ctl.enable_output(True)
            self.get_logger().warning('goal received: output ARMED')
        self.get_logger().info(
            f'goal {goal.label or goal.id}: ({goal.x:.1f}, {goal.y:.1f}, {goal.z:.1f})'
        )

    def _on_audio(self, msg) -> None:
        import numpy as np

        data = getattr(msg, 'data', None)
        if not data:
            return
        samples = np.frombuffer(bytes(data), dtype=np.int16)
        self._sound_db = self.ctl.audio.level_db(samples, 16000.0)

    # --- the control timer ---------------------------------------------------

    def _on_tick(self) -> None:
        now = self._now()
        with self._lock:
            frame, frame_t = self._frame, self._frame_t
            lidar_threats, lidar_t = self._lidar_threats, self._lidar_t
        boxes: list[Box] = []
        if frame is not None and now - frame_t <= FRAME_MAX_AGE_S:
            boxes = self.detector.detect(frame)
        hfov = self.ctl.tracker.hfov_deg
        threats = [threat_from_box(b, hfov, t=frame_t) for b in boxes]
        if lidar_threats and now - lidar_t <= LIDAR_MAX_AGE_S:
            threats += lidar_threats

        arousal = float(self.get_parameter('arousal').value)
        if arousal:
            self.ctl.motor.pursuit.arousal = arousal
        want_output = bool(self.get_parameter('enable_output').value)
        if want_output != self.ctl.output_enabled:
            self.ctl.enable_output(want_output)

        percept = self.ctl.tracker.update_threats(threats, now)
        self._consult_advisor(frame, percept, now)
        percept.altitude_m = self.ctl.state.altitude_m
        percept.arousal = self.ctl.motor.pursuit.arousal
        percept.sound_db = self._sound_level(percept)
        try:
            cmd = self.ctl.tick(percept, now=now)
        except Exception as exc:
            self.get_logger().error(f'tick failed: {type(exc).__name__}: {exc}')
            return
        threat = self.ctl.tracker.last
        if cmd.state in ('ESCAPE', 'AVOID'):
            seen = f' [{threat.source} #{threat.track_id} {threat.label}]' if threat else ''
            self.get_logger().warning(f'{cmd.state}: {cmd.reason}{seen}')
        status = self._String()
        status.data = self.ctl.goal_status().model_dump_json()
        self.status_pub.publish(status)
        self._publish_telemetry(now, percept, cmd, threat, len(threats))

    def _publish_telemetry(self, now: float, percept: Percept, cmd, threat, n_threats: int) -> None:
        out = self.ctl.last_out
        st = self.ctl.state
        rec = {
            't': round(now, 3),
            'state': cmd.state,
            'reason': cmd.reason,
            'cmd': [round(cmd.vx, 3), round(cmd.vy, 3), round(cmd.vz, 3), round(cmd.yaw_rate, 3)],
            'armed': self.ctl.output_enabled,
            'spikes': dict(out.spikes) if out is not None else {},
            'percept': {
                'has_target': percept.has_target,
                'bearing_deg': round(percept.bearing_deg, 1),
                'elevation_deg': round(percept.elevation_deg, 1),
                'size_deg': round(percept.size_deg, 2),
                'expansion_dps': round(percept.expansion_dps, 2),
                'distance_m': None if percept.distance_m is None else round(percept.distance_m, 2),
            },
            'threat': None if threat is None else {
                'source': threat.source, 'track_id': threat.track_id, 'label': threat.label,
            },
            'n_threats': n_threats,
            'lidar': {
                'scans': self._lidar_scans, 'points': self._lidar_points,
                'clusters': len(self.lidar.last_clusters),
                'threats': [
                    {'id': th.track_id, 'bearing': round(th.bearing_deg, 1),
                     'elev': round(th.elevation_deg, 1), 'size': round(th.size_deg, 2),
                     'range': None if th.distance_m is None else round(th.distance_m, 2),
                     'settled': th.settled, 'moving': th.moving,
                     'exp': round(self.ctl.tracker.expansion_of(th), 2)}
                    for th in sorted(self._lidar_threats, key=lambda x: x.distance_m or 0)[:4]
                ],
            },
            'position': [round(v, 2) for v in st.position],
            'yaw_rad': round(st.yaw_rad, 3),
            'goal_reached': self.ctl.motor.goal_reached,
            'dodge': None if self.ctl.motor.last_dodge is None else {
                'direction': self.ctl.motor.last_dodge.direction, 'source': self.ctl.motor.last_dodge.source,
                'expected': self.ctl.motor.last_dodge.expected,
            },
            'advisor': None if self.advisor is None else {
                'calls': self.advisor.calls, 'errors': self.advisor.errors, 'last_error': self.advisor.last_error,
            },
        }
        msg = self._String()
        msg.data = json.dumps(rec)
        self.telemetry_pub.publish(msg)

    def _publish_muse_raw(self, record: dict) -> None:
        """Publish one raw Muse answer as a JSON String, off the control path."""
        try:
            msg = self._String()
            msg.data = json.dumps(record, default=str)
            self.muse_raw_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warning(f'cannot publish Muse raw output: {exc}', once=True)

    def _consult_advisor(self, frame, percept: Percept, now: float) -> None:
        """Hand Muse the current threat (off the control path) and pass on any answer."""
        if self.advisor is None:
            return
        th = self.ctl.tracker.last
        if th is None or th.moving is False:
            return
        key = f'{th.source}:{th.track_id}'
        st = self.ctl.state
        self.advisor.consider(frame, {
            'label': th.label, 'bearing_deg': th.bearing_deg, 'elevation_deg': th.elevation_deg,
            'distance_m': th.distance_m, 'size_deg': th.size_deg, 'velocity_body': th.velocity_body,
            'speed_mps': math.hypot(st.velocity[0], st.velocity[1]),
        }, key, time.time())
        advice = self.advisor.latest(key)
        if advice is not None and advice is not self._last_advice:
            self._last_advice = advice
            self.ctl.set_dodge_advice(advice)
            self.get_logger().info(
                f'Muse: {advice.direction} for {key} ({advice.expected}; {advice.reason[:60]})'
            )

    # --- helpers -------------------------------------------------------------

    def _sound_level(self, percept: Percept) -> float:
        """
        Real level if an audio topic is feeding us, else inverse-square from the target.

        The synthetic fallback exists because the sound channel changes *when* the giant fibre
        fires (the connectome lowers the loom response under loud sound, and
        raises it on the toy graph), so the reflex cannot be characterised without it.
        """
        if self.get_parameter('audio_mode').value == 'topic' and HAVE_AUDIO:
            return self._sound_db
        if not percept.has_target or not percept.size_deg:
            return -80.0
        # a 1 m reference at 0 dB, falling 20 dB per decade of range; size stands in for range
        return max(-80.0, min(0.0, -40.0 + 20.0 * math.log10(max(percept.size_deg, 0.1))))

    def _to_cv(self, msg):
        try:
            from cv_bridge import CvBridge

            if self._bridge_cv is None:
                self._bridge_cv = CvBridge()
            return self._bridge_cv.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'cannot convert image: {exc}', once=True)
            return None

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def destroy_node(self) -> None:
        """Zero the vehicle on the way out, even if the stop itself misbehaves."""
        try:
            self.ctl.stop()
        except Exception as exc:
            self.get_logger().error(f'stop on shutdown failed: {exc}')
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrainDroneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
