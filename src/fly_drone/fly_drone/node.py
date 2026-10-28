"""
The rclpy node: sensors in, one brain tick per control period, velocity out.

Shape of a cycle::

    Image  --detect-->  boxes --LoomTracker--> Percept --encode--> spikes --motor--> Twist
    Odometry ----------------------------------^ (altitude, velocity)
    Audio   -----------------------------------^ (optional; synthesised if no audio topic)

**Output is off until something arms it.** The node starts with the controller's output gate
closed, so it can be run against a live simulator to watch what the brain would do before
anything is allowed to move. Arm it with the ``enable_output`` MCP tool or the
``~/enable_output`` parameter.

Why the brain does not run in the subscription callback: a tick costs tens of milliseconds
(see the measured ``run()`` overhead in ``REFLEX.md``), which would block the executor and
back up the image queue. It runs on a timer instead, on the latest frame only, and old frames
are dropped rather than queued.
"""
from __future__ import annotations

import math
import threading

from fly_brain.brain import preimport_flycns
from fly_brain.config import reload_settings
from fly_brain.controller import controller
from fly_brain.schemas import Box, Percept

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from .bridge import make_bridge
from .detect import Detector

try:                                    # not in this workspace's environment; optional
    from audio_common_msgs.msg import AudioData
    HAVE_AUDIO = True
except ImportError:
    AudioData = None
    HAVE_AUDIO = False


class BrainDroneNode(Node):
    """Wires the shared :class:`fly_brain.Controller` to ROS topics."""

    def __init__(self) -> None:
        super().__init__('fly_brain_pilot')
        self.cfg = reload_settings()
        self.ctl = controller()
        self.ctl.cfg = self.cfg

        self.declare_parameter('enable_output', False)
        self.declare_parameter('arousal', 0.0)
        self.declare_parameter('audio_mode', 'synthetic' if not HAVE_AUDIO else 'topic')

        self._lock = threading.Lock()
        self._frame = None               # latest image, BGR
        self._frame_t = 0.0
        self._sound_db = -80.0
        self.detector = Detector()
        self._bridge_cv = None

        # Brian2 installs its SIGINT handler at import and can only do that on the main
        # thread; rclpy callbacks are not it. Import before the first tick.
        if self.cfg.backend == 'flycns':
            if preimport_flycns(self.cfg.repo):
                self.get_logger().info(f'flycns imported from {self.cfg.repo}')
            else:
                self.get_logger().error(
                    f'flycns not importable from FLYDRONE_REPO={self.cfg.repo!r}; '
                    'the brain will fail to load. Falling back to BRAIN_BACKEND=mock is safe.'
                )

        self.bridge = make_bridge(
            self, self.cfg.ros_bridge, self.cfg.topic_cmd_vel, self.cfg.frame_id
        )
        self.ctl.set_bridge(self.bridge)
        self.ctl.enable_output(bool(self.get_parameter('enable_output').value))

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image

        self.create_subscription(Odometry, self.cfg.topic_odom, self._on_odom, sensor_qos)
        self.create_subscription(Image, self.cfg.topic_image, self._on_image, sensor_qos)
        if HAVE_AUDIO and self.get_parameter('audio_mode').value == 'topic':
            self.create_subscription(AudioData, self.cfg.topic_audio, self._on_audio, sensor_qos)

        period = 1.0 / max(self.cfg.loop_hz, 0.1)
        self.timer = self.create_timer(period, self._on_tick)
        self.get_logger().info(
            f'fly_brain_pilot up: backend={self.cfg.backend} bridge={self.bridge.name} '
            f'cmd={self.cfg.topic_cmd_vel} odom={self.cfg.topic_odom} '
            f'image={self.cfg.topic_image} '
            f'loop={self.cfg.loop_hz} Hz tick={self.cfg.brain_tick_ms} ms '
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
        # yaw from the quaternion's z/w terms only; roll and pitch are small in level flight
        st.yaw_rad = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))

    def _on_image(self, msg) -> None:
        """Keep only the newest frame: the brain cannot keep up with the camera."""
        img = self._to_cv(msg)
        if img is None:
            return
        with self._lock:
            self._frame = img
            self._frame_t = self._now()

    def _on_audio(self, msg) -> None:
        import numpy as np

        data = getattr(msg, 'data', None)
        if not data:
            return
        samples = np.frombuffer(bytes(data), dtype=np.int16)
        self._sound_db = self.ctl.audio.level_db(samples, 16000.0)

    # --- the control timer ---------------------------------------------------

    def _on_tick(self) -> None:
        with self._lock:
            frame, frame_t = self._frame, self._frame_t
        boxes: list[Box] = self.detector.detect(frame) if frame is not None else []

        arousal = float(self.get_parameter('arousal').value)
        if arousal:
            self.ctl.motor.pursuit.arousal = arousal
        want_output = bool(self.get_parameter('enable_output').value)
        if want_output != self.ctl.output_enabled:
            self.ctl.enable_output(want_output)

        percept = self.ctl.tracker.update(boxes, frame_t or self._now())
        percept.altitude_m = self.ctl.state.altitude_m
        percept.arousal = self.ctl.motor.pursuit.arousal
        percept.sound_db = self._sound_level(percept)
        try:
            cmd = self.ctl.tick(percept)
        except Exception as exc:
            self.get_logger().error(f'tick failed: {type(exc).__name__}: {exc}')
            return
        if cmd.state in ('ESCAPE', 'AVOID'):
            self.get_logger().warning(f'{cmd.state}: {cmd.reason}')

    # --- helpers -------------------------------------------------------------

    def _sound_level(self, percept: Percept) -> float:
        """
        Real level if an audio topic is feeding us, else inverse-square from the target.

        The synthetic fallback exists because the sound channel changes *when* the giant fibre
        fires (FLYCNS reports sound lowers the loom response on the real connectome, and
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
