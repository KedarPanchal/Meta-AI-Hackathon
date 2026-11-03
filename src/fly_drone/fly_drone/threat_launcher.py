"""
The threat launcher node: spawns objects on a collision course with the drone, in Gazebo.

Why it exists: the escape reflex cannot be demonstrated, let alone calibrated, in a forest
where nothing moves. This node reads the drone's odometry and, on a schedule or on request,
spawns a small sphere ``distance_m`` away at a chosen bearing and elevation and throws it at
the drone at ``speed_mps`` (see :mod:`fly_drone.threats`). The brain sees it on the forward
camera and on the lidar, the giant fibre fires, and the drone dodges; the object is removed
again after its time of flight.

The sphere is moved **kinematically**, 20 times per simulated second, through Gazebo's
``set_pose`` service; the ``create``, ``remove`` and ``set_pose`` services reach ROS through
``ros_gz_bridge`` (``brain_launch.py`` starts that bridge). Simulation time, so a slow
rendering world still sees the sphere at its commanded speed in its own clock.

Modes (``mode`` parameter):

* ``periodic`` -- one launch every ``period_s`` after ``delay_s``, up to ``max_launches``
  (0: no limit). A ``period_s`` of 0 means nothing automatic at all: ``topic`` mode.
* ``once`` -- a single launch after ``delay_s``.
* ``topic`` -- only on request.

In every mode a ``std_msgs/String`` on ``/threats/launch`` launches one immediately; the
string may be empty or a JSON object overriding any of ``distance_m``, ``speed_mps``,
``bearing_deg``, ``elevation_deg``, ``miss_m``, ``radius_m``, ``color``, ``lead``. Every spawn,
(five times a second) position and the **result** is reported as JSON on ``/threats/events``
so a test can line events up against the brain's telemetry.

**Did it hit?** You cannot tell from the GUI whether the drone passed under the sphere or
through it, so the launcher keeps the closest distance between the two centres over the
sphere's flight and, when it takes the sphere away, logs ``HIT`` or ``MISS`` with that
distance, counts them, and leaves a self-lit **red ball** where the sphere was at its
closest for eight seconds when it hit. (``mark_misses:=true`` leaves a green one for misses
too; off by default, since a miss is the normal case and the balls clutter the scene.)
"""
from __future__ import annotations

import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from .threats import Launch, aim, marker_sdf, threat_sdf

DRONE_RADIUS_M = 0.35
"""Half the X3's 0.47 m frame plus a rotor's reach, roughly: a sphere centre within this plus
its own radius has touched the airframe."""
MARKER_TTL_S = 8.0

POSE_HZ = 20.0
REPORT_EVERY = 4
"""Position events every this many pose updates: five a second."""


class ThreatLauncher(Node):
    def __init__(self) -> None:
        super().__init__('threat_launcher')
        self.declare_parameter('world', 'forest_search')
        self.declare_parameter('mode', 'periodic')
        self.declare_parameter('delay_s', 15.0)
        self.declare_parameter('period_s', 20.0)
        self.declare_parameter('max_launches', 0)
        self.declare_parameter('distance_m', 10.0)
        self.declare_parameter('speed_mps', 5.0)
        self.declare_parameter('bearing_deg', 0.0)
        self.declare_parameter('elevation_deg', 0.0)
        self.declare_parameter('miss_m', 0.0)
        self.declare_parameter('radius_m', 0.25)
        self.declare_parameter('color', 'magenta')
        self.declare_parameter('lead', True)
        self.declare_parameter('odom_topic', '/drone/odometry')
        self.declare_parameter('mark_misses', False)

        from nav_msgs.msg import Odometry
        from ros_gz_interfaces.srv import DeleteEntity, SetEntityPose, SpawnEntity
        from std_msgs.msg import String

        self._String = String
        self._SpawnEntity, self._DeleteEntity, self._SetEntityPose = (
            SpawnEntity, DeleteEntity, SetEntityPose
        )
        world = str(self.get_parameter('world').value)
        self.world = world
        self.spawn_cli = self.create_client(SpawnEntity, f'/world/{world}/create')
        self.remove_cli = self.create_client(DeleteEntity, f'/world/{world}/remove')
        self.pose_cli = self.create_client(SetEntityPose, f'/world/{world}/set_pose')

        self._pose: tuple[float, float, float] | None = None
        self._yaw = 0.0
        self._vel_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._live: list[Launch] = []
        self._markers: list[tuple[float, str]] = []           # (remove at sim time, name)
        self.hits = 0
        self.misses = 0
        self._n = 0
        self._updates = 0
        self._next_at: float | None = None
        mode = str(self.get_parameter('mode').value)
        if mode == 'periodic' and float(self.get_parameter('period_s').value) <= 0.0:
            mode = 'topic'                      # period 0: nothing automatic, only on request
        self.mode = mode
        if mode in ('periodic', 'once'):
            self._next_at = time.monotonic() + float(self.get_parameter('delay_s').value)

        self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value), self._on_odom,
            QoSPresetProfiles.SENSOR_DATA.value,
        )
        self.create_subscription(String, '/threats/launch', self._on_request, 10)
        self.events = self.create_publisher(String, '/threats/events', 10)
        self.create_timer(1.0 / POSE_HZ, self._tick)
        self.get_logger().info(
            f'threat launcher up: world={world} mode={mode} '
            f'distance={self.get_parameter("distance_m").value} m '
            f'speed={self.get_parameter("speed_mps").value} m/s; '
            'publish to /threats/launch to throw one now'
        )

    # --- inputs -----------------------------------------------------------------

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_odom(self, msg) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        # the odometry twist is in the body frame; the aim wants world velocity
        c, s = math.cos(yaw), math.sin(yaw)
        self._pose = (p.x, p.y, p.z)
        self._yaw = yaw
        self._vel_world = (c * v.x - s * v.y, s * v.x + c * v.y, v.z)

    def _on_request(self, msg) -> None:
        overrides: dict = {}
        text = (msg.data or '').strip()
        if text:
            try:
                overrides = json.loads(text)
            except json.JSONDecodeError as exc:
                self.get_logger().error(f'bad launch request {text!r}: {exc}')
                return
        self.launch(overrides)

    # --- the schedule and the motion ----------------------------------------------

    def _tick(self) -> None:
        now = self._now()
        wall = time.monotonic()
        self._updates += 1
        for launch in list(self._live):
            if now - launch.t0 >= launch.ttl_s:
                self._live.remove(launch)
                self._finish(launch)
                continue
            pos = launch.position_at(now)
            if self._pose is not None:
                launch.note(pos, self._pose, now)
            req = self._SetEntityPose.Request()
            req.entity.name = launch.name
            req.entity.type = req.entity.MODEL
            req.pose.position.x, req.pose.position.y, req.pose.position.z = pos
            req.pose.orientation.w = 1.0
            self.pose_cli.call_async(req)
            if self._updates % REPORT_EVERY == 0:
                self._event('pose', launch, ok=True, position=pos)
        if self._next_at is not None and wall >= self._next_at:
            if self.launch({}):
                limit = int(self.get_parameter('max_launches').value)
                period = float(self.get_parameter('period_s').value)
                if self.mode == 'periodic' and period > 0.0 and (limit <= 0 or self._n < limit):
                    self._next_at = wall + period
                else:
                    self._next_at = None
            else:
                self._next_at = wall + 1.0        # no odometry yet, or no service: retry

    def _finish(self, launch: Launch) -> None:
        """Take the sphere away, say whether it hit, and leave a marker where it came closest."""
        req = self._DeleteEntity.Request()
        req.entity.name = launch.name
        req.entity.type = req.entity.MODEL
        self.remove_cli.call_async(req)
        hit = launch.hit(DRONE_RADIUS_M)
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        verdict = 'HIT' if hit else 'MISS'
        self.get_logger().warning(
            f'{launch.name}: {verdict} -- closest {launch.closest_m:.2f} m between centres '
            f'(touch at {launch.radius_m + DRONE_RADIUS_M:.2f} m), {self.hits} hit / {self.misses} missed so far'
        )
        self._event('result', launch, ok=not hit, position=launch.closest_at,
                    extra={'hit': hit, 'closest_m': round(launch.closest_m, 2),
                           'hits': self.hits, 'misses': self.misses})
        if not hit and not bool(self.get_parameter('mark_misses').value):
            return                                  # a miss is the normal case: log only
        name = f'{launch.name}_{verdict.lower()}'
        mk = self._SpawnEntity.Request()
        mk.entity_factory.name = name
        mk.entity_factory.sdf = marker_sdf(name, hit)
        (mk.entity_factory.pose.position.x, mk.entity_factory.pose.position.y,
         mk.entity_factory.pose.position.z) = launch.closest_at
        mk.entity_factory.pose.orientation.w = 1.0
        self.spawn_cli.call_async(mk)
        self._markers.append((self._now() + MARKER_TTL_S, name))
        now = self._now()
        for expires, old in list(self._markers):
            if now >= expires:
                self._markers.remove((expires, old))
                rm = self._DeleteEntity.Request()
                rm.entity.name = old
                rm.entity.type = rm.entity.MODEL
                self.remove_cli.call_async(rm)

    def launch(self, overrides: dict) -> bool:
        if self._pose is None:
            self.get_logger().warning('no odometry yet; cannot aim', throttle_duration_sec=5.0)
            return False
        if not self.spawn_cli.service_is_ready():
            self.get_logger().warning(
                f'/world/{self.world}/create is not bridged yet; is the service bridge up?',
                throttle_duration_sec=5.0,
            )
            return False

        def p(name: str, cast=float):
            return cast(overrides.get(name, self.get_parameter(name).value))

        distance, speed = p('distance_m'), p('speed_mps')
        bearing, elevation = p('bearing_deg'), p('elevation_deg')
        miss, radius = p('miss_m'), p('radius_m')
        color, lead = p('color', str), p('lead', bool)
        pos, vel, tof = aim(
            self._pose, self._yaw, self._vel_world, distance, bearing, elevation, speed,
            miss_m=miss, lead=lead,
        )
        self._n += 1
        name = f'threat_{self._n}'
        launch = Launch(
            name=name, position=pos, velocity=vel, radius_m=radius, color=color,
            ttl_s=tof + 2.5, distance_m=distance, bearing_deg=bearing,
            elevation_deg=elevation, speed_mps=speed, tof_s=tof, t0=self._now(),
        )
        req = self._SpawnEntity.Request()
        req.entity_factory.name = name
        req.entity_factory.allow_renaming = False
        req.entity_factory.sdf = threat_sdf(name, radius, color)
        req.entity_factory.pose.position.x, req.entity_factory.pose.position.y, \
            req.entity_factory.pose.position.z = pos
        req.entity_factory.pose.orientation.w = 1.0
        fut = self.spawn_cli.call_async(req)
        fut.add_done_callback(lambda f, la=launch: self._spawned(la, f))
        self.get_logger().warning(
            f'{name}: {radius * 2:.2f} m {color} sphere from {distance:.1f} m at bearing '
            f'{bearing:+.0f} deg, elevation {elevation:+.0f} deg, closing at {speed:.1f} m/s '
            f'(contact in {tof:.1f} s)'
        )
        return True

    def _spawned(self, launch: Launch, fut) -> None:
        ok = self._ok(fut)
        if ok:
            launch.t0 = self._now()                # the clock starts when Gazebo has it
            self._live.append(launch)
            self._event('spawned', launch, ok=True)
        else:
            self.get_logger().error(f'spawn of {launch.name} failed')
            self._event('spawn_failed', launch, ok=False)

    @staticmethod
    def _ok(fut) -> bool:
        try:
            res = fut.result()
        except Exception:
            return False
        return bool(getattr(res, 'success', False))

    def _event(self, kind: str, launch: Launch, ok: bool, position=None, extra: dict | None = None) -> None:
        msg = self._String()
        msg.data = json.dumps({
            **(extra or {}),
            'event': kind, 'ok': ok, 't': time.time(), 'sim_t': self._now(),
            'name': launch.name,
            'position': list(position) if position is not None else list(launch.position),
            'velocity': list(launch.velocity),
            'distance_m': launch.distance_m, 'bearing_deg': launch.bearing_deg,
            'elevation_deg': launch.elevation_deg, 'speed_mps': launch.speed_mps,
            'radius_m': launch.radius_m, 'color': launch.color, 'tof_s': round(launch.tof_s, 2),
            'drone_position': list(self._pose) if self._pose else None,
            'drone_yaw_rad': self._yaw,
        })
        self.events.publish(msg)

    def destroy_node(self) -> None:
        for launch in self._live:
            req = self._DeleteEntity.Request()
            req.entity.name = launch.name
            req.entity.type = req.entity.MODEL
            self.remove_cli.call_async(req)
        self._live.clear()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ThreatLauncher()
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
