"""
Vehicle interfaces. One of these carries the brain's command to the aircraft.

The default is :class:`TwistBridge`, a body-frame ``geometry_msgs/Twist``, because that is
what plain Gazebo gives you through ``ros_gz_bridge`` and it is the only interface this
workspace actually has the message packages for. The PX4 and MAVROS adapters are written out
but import their message packages lazily, so selecting one without installing it fails with a
sentence telling you what to add rather than an ImportError at module load.

Body frame is FLU and ``geometry_msgs/Twist`` is already FLU, so ``Command`` maps onto it
field for field with no sign changes: ``linear.x`` forward, ``linear.y`` left, ``linear.z``
up, ``angular.z`` yaw counter-clockwise.
"""
from __future__ import annotations

from fly_brain.schemas import Command


class TwistBridge:
    """``geometry_msgs/Twist`` on a velocity topic: plain Gazebo through ros_gz_bridge."""

    name = 'twist'

    def __init__(self, node, topic: str, stamped: bool = False, frame_id: str = 'base_link'):
        from geometry_msgs.msg import Twist, TwistStamped

        self.node = node
        self.stamped = stamped
        self.frame_id = frame_id
        self._Twist = Twist
        self._TwistStamped = TwistStamped
        self.pub = node.create_publisher(TwistStamped if stamped else Twist, topic, 10)
        self.topic = topic
        self.sent = 0

    def send(self, cmd: Command) -> None:
        msg = self._Twist()
        msg.linear.x = float(cmd.vx)
        msg.linear.y = float(cmd.vy)
        msg.linear.z = float(cmd.vz)
        msg.angular.z = float(cmd.yaw_rate)
        if self.stamped:
            wrapper = self._TwistStamped()
            wrapper.header.stamp = self.node.get_clock().now().to_msg()
            wrapper.header.frame_id = self.frame_id
            wrapper.twist = msg
            self.pub.publish(wrapper)
        else:
            self.pub.publish(msg)
        self.sent += 1

    def stop(self) -> None:
        self.send(Command())


class Px4Bridge:
    """
    PX4 offboard velocity through uXRCE-DDS (``px4_msgs``).

    PX4 setpoints are in NED, not FLU, and PX4 wants a steady stream of
    ``OffboardControlMode`` messages or it drops out of offboard, both of which are handled
    here. Not exercised: ``px4_msgs`` is not in this workspace's pixi environment.
    """

    name = 'px4'

    def __init__(self, node, topic: str = '/fmu/in/trajectory_setpoint'):
        try:
            from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint
        except ImportError as exc:                       # pragma: no cover - px4_msgs absent
            raise RuntimeError(
                'ROS_BRIDGE=px4 needs the px4_msgs package. Add it to pixi.toml (it is not in '
                'this environment) and rebuild, or use ROS_BRIDGE=twist.'
            ) from exc
        self.node = node
        self._Setpoint = TrajectorySetpoint
        self._Mode = OffboardControlMode
        self.pub = node.create_publisher(TrajectorySetpoint, topic, 10)
        self.mode_pub = node.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10
        )
        self.sent = 0

    def _stamp(self) -> int:
        return int(self.node.get_clock().now().nanoseconds / 1000)

    def send(self, cmd: Command) -> None:
        mode = self._Mode()
        mode.timestamp = self._stamp()
        mode.position = False
        mode.velocity = True
        self.mode_pub.publish(mode)

        sp = self._Setpoint()
        sp.timestamp = self._stamp()
        # FLU (x fwd, y left, z up) -> NED (x fwd, y right, z down)
        sp.velocity = [float(cmd.vx), float(-cmd.vy), float(-cmd.vz)]
        sp.position = [float('nan')] * 3
        sp.acceleration = [float('nan')] * 3
        sp.yawspeed = float(-cmd.yaw_rate)    # NED yaw is clockwise-positive
        self.pub.publish(sp)
        self.sent += 1

    def stop(self) -> None:
        self.send(Command())


class MavrosBridge:
    """
    MAVROS ``setpoint_velocity`` (``geometry_msgs/TwistStamped``, body frame).

    Not exercised: ``mavros_msgs`` is not in this workspace's pixi environment, though this
    adapter only needs ``geometry_msgs`` once MAVROS itself is running.
    """

    name = 'mavros'

    def __init__(self, node, topic: str = '/mavros/setpoint_velocity/cmd_vel'):
        from geometry_msgs.msg import TwistStamped

        self.node = node
        self._TwistStamped = TwistStamped
        self.pub = node.create_publisher(TwistStamped, topic, 10)
        self.sent = 0

    def send(self, cmd: Command) -> None:
        msg = self._TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(cmd.vx)
        msg.twist.linear.y = float(cmd.vy)
        msg.twist.linear.z = float(cmd.vz)
        msg.twist.angular.z = float(cmd.yaw_rate)
        self.pub.publish(msg)
        self.sent += 1

    def stop(self) -> None:
        self.send(Command())


def make_bridge(node, kind: str, topic: str, frame_id: str = 'base_link'):
    """Build whichever bridge ``ROS_BRIDGE`` names."""
    kind = (kind or 'twist').lower()
    if kind == 'twist':
        return TwistBridge(node, topic, stamped=False, frame_id=frame_id)
    if kind in ('twist_stamped', 'twiststamped'):
        return TwistBridge(node, topic, stamped=True, frame_id=frame_id)
    if kind == 'px4':
        return Px4Bridge(
            node, topic if topic.startswith('/fmu') else '/fmu/in/trajectory_setpoint'
        )
    if kind == 'mavros':
        return MavrosBridge(
            node, topic if 'mavros' in topic else '/mavros/setpoint_velocity/cmd_vel'
        )
    raise ValueError(f'unknown ROS_BRIDGE {kind!r}; choose twist, twist_stamped, px4 or mavros')
