"""
How the mission layer reaches the brain. Two implementations of one small protocol.

* :class:`RosLink` -- the real one. A ``mission_link`` rclpy node spun on a daemon thread:
  goals go out as JSON on ``/fly_brain/goal``, and odometry, the downward camera and the
  brain's ``GoalStatus`` come back. The brain runs in its own process (``fly_brain_pilot``).
* :class:`SimLink` -- the brain *in this process* driving a kinematic drone over a flat world
  with one person on it, rendered into a synthetic downward camera. It exercises the real
  ``MotorLayer`` navigation, the real planner and the real runner without Gazebo or ROS, so the
  mission can be tested end to end on a laptop, and it is the ``auto`` fallback when rclpy is
  not importable.

The protocol is deliberately tiny: the mission layer says where, reads where it is and what it
sees, and nothing else. Velocity is the brain's.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Protocol, runtime_checkable

from fly_brain.schemas import Command, Goal, GoalStatus, Percept

from .camera import DownCamera
from .config import MissionSettings


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw: float
    t: float
    """Wall clock when it was measured, s since epoch."""

    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Frame:
    image: Any
    """HxWx3 uint8 BGR numpy array."""
    t: float


@runtime_checkable
class BrainLink(Protocol):
    name: str

    def goto(self, goal: Goal) -> None: ...

    def pose(self) -> Pose | None: ...

    def goal_status(self) -> GoalStatus | None: ...

    def frame(self) -> Frame | None: ...

    def describe(self) -> dict: ...

    def close(self) -> None: ...


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def image_msg_to_bgr(msg):
    """sensor_msgs/Image to a BGR array for the common encodings; None if it cannot be done."""
    import numpy as np

    enc = (msg.encoding or '').lower()
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    channels = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4, 'mono8': 1, '8uc1': 1, '8uc3': 3}.get(enc)
    if channels is None or buf.size < h * step:
        try:
            from cv_bridge import CvBridge

            return CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return None
    rows = buf[: h * step].reshape(h, step)[:, : w * channels].reshape(h, w, channels)
    if enc == 'rgb8':
        return np.ascontiguousarray(rows[:, :, ::-1])
    if enc == 'rgba8':
        return np.ascontiguousarray(rows[:, :, [2, 1, 0]])
    if enc == 'bgra8':
        return np.ascontiguousarray(rows[:, :, :3])
    if channels == 1:
        return np.ascontiguousarray(np.repeat(rows, 3, axis=2))
    return np.ascontiguousarray(rows)


class RosLink:
    """Topics to and from the ``fly_brain_pilot`` node in another process."""

    name = 'ros'

    def __init__(self, cfg: MissionSettings):
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSPresetProfiles
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        self.cfg = cfg
        b = cfg.brain
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node('mission_link')
        self._String = String
        self.pub_goal = self.node.create_publisher(String, b.topic_goal, 10)
        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.node.create_subscription(Odometry, b.topic_odom, self._on_odom, qos)
        self.node.create_subscription(Image, b.topic_image_down, self._on_image, qos)
        self.node.create_subscription(String, b.topic_goal_status, self._on_status, 10)
        self._lock = threading.Lock()
        self._pose: Pose | None = None
        self._frame: Frame | None = None
        self._status: GoalStatus | None = None
        self.last_goal: Goal | None = None
        self.goals_sent = 0
        self.frames = 0
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self.node)
        self._thread = threading.Thread(target=self._spin, name='mission-link-spin', daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        try:
            self._exec.spin()
        except Exception:
            pass

    def _on_odom(self, msg) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        pose = Pose(p.x, p.y, p.z, yaw_from_quaternion(q.x, q.y, q.z, q.w), time.time())
        with self._lock:
            self._pose = pose

    def _on_image(self, msg) -> None:
        img = image_msg_to_bgr(msg)
        if img is None:
            return
        with self._lock:
            self._frame = Frame(img, time.time())
            self.frames += 1

    def _on_status(self, msg) -> None:
        try:
            st = GoalStatus.model_validate_json(msg.data)
        except Exception:
            return
        with self._lock:
            self._status = st

    def goto(self, goal: Goal) -> None:
        m = self._String()
        m.data = goal.model_dump_json()
        self.pub_goal.publish(m)
        self.last_goal = goal
        self.goals_sent += 1

    def pose(self) -> Pose | None:
        with self._lock:
            return self._pose

    def goal_status(self) -> GoalStatus | None:
        with self._lock:
            return self._status

    def frame(self) -> Frame | None:
        with self._lock:
            return self._frame

    def describe(self) -> dict:
        b = self.cfg.brain
        with self._lock:
            return {
                'link': self.name,
                'goal_topic': b.topic_goal, 'goal_status_topic': b.topic_goal_status,
                'odom_topic': b.topic_odom, 'image_topic': b.topic_image_down,
                'have_odom': self._pose is not None, 'have_frame': self._frame is not None,
                'have_brain_status': self._status is not None,
                'goals_sent': self.goals_sent, 'frames': self.frames,
            }

    def close(self) -> None:
        try:
            self._exec.shutdown(timeout_sec=1.0)
            self.node.destroy_node()
        except Exception:
            pass


class SimLink:
    """
    The brain in-process, flying a kinematic drone over a flat world with one person on it.

    Each step feeds the controller the simulated pose, runs one real brain tick (so NAVIGATE,
    the reflexes and the safety clamp are all the production code), then integrates the
    body-frame command with a first-order lag that stands in for the multicopter's response.
    The camera renders the ground as green and the person as a magenta figure, which is what
    :class:`mission.detector.ColorDetector` looks for. The planner is never told where the
    person is; only the renderer knows.
    """

    name = 'sim'

    PERSON_SIZE_M = (1.9, 0.6)
    PERSON_BGR = (255, 0, 255)
    GROUND_BGR = (40, 95, 35)

    def __init__(
        self,
        cfg: MissionSettings,
        person_xy: tuple[float, float] | None = None,
        start: tuple[float, float, float] | None = None,
        speedup: float | None = None,
        dt: float | None = None,
        camera: DownCamera | None = None,
        velocity_tau_s: float = 0.4,
    ):
        from fly_brain.controller import controller

        self.cfg = cfg
        self.camera = camera or DownCamera(cfg.cam_hfov_deg, cfg.cam_width, cfg.cam_height)
        self.person_xy = tuple(person_xy or cfg.sim_person_xy)
        self.person_heading = 0.35
        x, y, z = start or cfg.sim_start
        self.x, self.y, self.z, self.yaw = float(x), float(y), float(z), 0.0
        self.vx = self.vy = self.vz = 0.0
        self.floor_z = min(float(z), cfg.ground_z_m)
        self.ground_z = cfg.ground_z_m
        self.dt = dt or cfg.sim_dt
        self.speedup = speedup or cfg.sim_speedup
        self.tau = velocity_tau_s
        self.ctl = controller()
        self.sim_time = 0.0
        self.distance_m = 0.0
        self.steps = 0
        self.frames = 0
        self.goals_sent = 0
        self.last_goal: Goal | None = None
        self.last_cmd: Command | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._push_state()
        self._thread = threading.Thread(target=self._run, name='mission-sim', daemon=True)
        self._thread.start()

    # --- the world ---------------------------------------------------------------

    def _push_state(self) -> None:
        st = self.ctl.state
        st.connected = True
        st.position = (self.x, self.y, self.z)
        st.velocity = (self.vx, self.vy, self.vz)
        st.yaw_rad = self.yaw
        st.altitude_m = self.z

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self.step()
            except Exception:
                pass
            budget = self.dt / max(self.speedup, 1e-3)
            rest = budget - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)

    def step(self) -> Command:
        """One brain tick and one integration step."""
        with self._lock:
            self._push_state()
            cmd = self.ctl.tick(Percept(has_target=False, altitude_m=self.z))
            if not self.ctl.output_enabled:
                cmd = Command(state=cmd.state, reason=cmd.reason + ' [disarmed]')
            self.last_cmd = cmd
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            wx = c * cmd.vx - s * cmd.vy
            wy = s * cmd.vx + c * cmd.vy
            a = self.dt / (self.tau + self.dt)
            self.vx += (wx - self.vx) * a
            self.vy += (wy - self.vy) * a
            self.vz += (cmd.vz - self.vz) * a
            self.x += self.vx * self.dt
            self.y += self.vy * self.dt
            self.z = max(self.floor_z, self.z + self.vz * self.dt)
            self.yaw = (self.yaw + cmd.yaw_rate * self.dt + math.pi) % (2 * math.pi) - math.pi
            self.distance_m += math.hypot(self.vx, self.vy) * self.dt
            self.sim_time += self.dt
            self.steps += 1
            return cmd

    # --- the protocol ----------------------------------------------------------------

    def goto(self, goal: Goal) -> None:
        with self._lock:
            self.ctl.set_goal(goal)
            if not self.ctl.output_enabled:
                self.ctl.enable_output(True)
            self.last_goal = goal
            self.goals_sent += 1

    def pose(self) -> Pose | None:
        with self._lock:
            return Pose(self.x, self.y, self.z, self.yaw, time.time())

    def goal_status(self) -> GoalStatus | None:
        return self.ctl.goal_status()

    def frame(self) -> Frame | None:
        with self._lock:
            img = self.render()
            self.frames += 1
            return Frame(img, time.time())

    def render(self):
        """The downward camera: flat green ground, one magenta person if in view."""
        import numpy as np

        h, w = self.camera.height, self.camera.width
        img = np.empty((h, w, 3), dtype=np.uint8)
        img[:, :] = self.GROUND_BGR
        drone = (self.x, self.y, self.z)
        length, width = self.PERSON_SIZE_M
        ch, sh = math.cos(self.person_heading), math.sin(self.person_heading)
        corners = []
        for dx, dy in ((length / 2, width / 2), (-length / 2, width / 2),
                       (-length / 2, -width / 2), (length / 2, -width / 2)):
            gx = self.person_xy[0] + dx * ch - dy * sh
            gy = self.person_xy[1] + dx * sh + dy * ch
            uv = self.camera.ground_to_pixel((gx, gy), drone, self.yaw, self.ground_z, clip=False)
            assert uv is not None
            corners.append((uv[0] * w, uv[1] * h))
        if all(c[0] < 0 or c[0] > w or c[1] < 0 or c[1] > h for c in corners):
            return img
        try:
            import cv2

            pts = np.array([[int(round(u)), int(round(v))] for u, v in corners], dtype=np.int32)
            cv2.fillPoly(img, [pts], self.PERSON_BGR)
        except ImportError:                              # pragma: no cover
            us = [c[0] for c in corners]
            vs = [c[1] for c in corners]
            u0, u1 = max(0, int(min(us))), min(w, int(max(us)) + 1)
            v0, v1 = max(0, int(min(vs))), min(h, int(max(vs)) + 1)
            img[v0:v1, u0:u1] = self.PERSON_BGR
        return img

    def describe(self) -> dict:
        with self._lock:
            return {
                'link': self.name, 'sim_time_s': round(self.sim_time, 1), 'speedup': self.speedup,
                'steps': self.steps, 'frames': self.frames, 'goals_sent': self.goals_sent,
                'distance_m': round(self.distance_m, 1), 'brain': self.ctl.status().backend,
                'last_command': self.last_cmd.model_dump() if self.last_cmd else None,
                'note': 'kinematic stand-in: flat ground, no obstacles, one person for the camera',
            }

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def make_link(cfg: MissionSettings) -> BrainLink:
    """``MISSION_LINK``: ros | sim | auto (ros if rclpy imports, else sim)."""
    kind = (cfg.link or 'auto').lower()
    if kind == 'auto':
        try:
            import rclpy  # noqa: F401

            kind = 'ros'
        except ImportError:
            kind = 'sim'
    if kind == 'ros':
        return RosLink(cfg)
    if kind == 'sim':
        return SimLink(cfg)
    raise ValueError(f'unknown MISSION_LINK {kind!r}; choose ros, sim or auto')
