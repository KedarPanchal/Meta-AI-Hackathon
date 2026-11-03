#!/usr/bin/env python3
"""Point the Gazebo GUI camera at the X3 UAV.

Gazebo's ``CameraTracking`` GUI plugin is loaded by the default GUI layout, but it starts in
track mode ``NONE``: it only moves the user camera once it receives a
``gz.msgs.CameraTrack`` message on ``/gui/track``. The GUI's "Camera Tracking" dialog and the
entity tree's "Follow" action both send exactly that message. There is no service for
following and ``ros_gz_bridge`` has no mapping for ``gz.msgs.CameraTrack``, so this node asks
Gazebo's own CLI to publish it instead of speaking gz-transport directly.

It lives in ``fly_bringup`` because it is only needed for the simulation launch; the world,
the GUI and this command are one setup.

The command is re-sent every ``rate_hz`` rather than once: the GUI may still be loading when
the node starts, the model can be respawned, and pressing Esc in the GUI drops the follow.
Re-sending the command for the target already being followed only re-applies the same offset
and gain.

Parameters
----------
target : str
    Name of the model to follow, as Gazebo knows it (the world's ``<include><name>``).
offset : list[float]
    Camera offset from the target in the target's local frame: negative x is behind the
    drone, positive z is above it.
pgain : float
    Follow proportional gain; small values lag, large values snap.
rate_hz : float
    How often the follow command is re-sent.
gz_executable : str
    The ``gz`` command to call, resolved on PATH.
"""
from __future__ import annotations

import shutil
import subprocess

import rclpy
from rclpy.node import Node

TRACK_TOPIC = '/gui/track'
TRACK_TYPE = 'gz.msgs.CameraTrack'


class FollowCamera(Node):
    """Publishes the CameraTrack command that makes the GUI camera follow a model."""

    def __init__(self) -> None:
        super().__init__('camera_tracking')
        self.declare_parameter('target', 'x3_search_uav')
        self.declare_parameter('offset', [-6.0, 0.0, 3.0])
        self.declare_parameter('pgain', 0.01)
        self.declare_parameter('rate_hz', 1.0)
        self.declare_parameter('gz_executable', 'gz')

        self.target = str(self.get_parameter('target').value).strip()
        self.offset = self._read_offset(self.get_parameter('offset').value)
        self.pgain = float(self.get_parameter('pgain').value)
        self.rate_hz = max(float(self.get_parameter('rate_hz').value), 0.1)
        self.gz = shutil.which(str(self.get_parameter('gz_executable').value))

        if not self.target:
            raise ValueError('target parameter is empty')
        if self.gz is None:
            raise RuntimeError(
                'the gz command line tool is not on PATH; start this node inside the '
                'pixi environment that provides Gazebo'
            )

        self.command = self._command()
        self._last_warning = None
        self.create_timer(1.0 / self.rate_hz, self._publish)
        self.get_logger().info(
            f'following {self.target} with the Gazebo camera: offset {self.offset}, '
            f'p-gain {self.pgain}, re-sent at {self.rate_hz:g} Hz on {TRACK_TOPIC}'
        )
        self._publish()

    def _read_offset(self, value) -> list[float]:
        """Accept the parameter as a double array or as one string of three numbers."""
        if isinstance(value, str):
            parts = value.replace(',', ' ').split()
        else:
            parts = list(value) if value is not None else []
        if len(parts) != 3:
            raise ValueError(f'offset needs three numbers, got {value!r}')
        return [float(part) for part in parts]

    def _command(self) -> list[str]:
        """The CLI call that publishes one follow command; ``-p`` takes protobuf text."""
        x, y, z = self.offset
        message = (
            'track_mode: FOLLOW\n'
            f'follow_target {{ name: "{self.target}" }}\n'
            f'follow_offset {{ x: {x} y: {y} z: {z} }}\n'
            f'follow_pgain: {self.pgain}\n'
        )
        return [self.gz, 'topic', '-t', TRACK_TOPIC, '-m', TRACK_TYPE, '-p', message]

    def _publish(self) -> None:
        try:
            done = subprocess.run(self.command, capture_output=True, text=True, timeout=5.0)
        except (OSError, subprocess.SubprocessError) as exc:
            self._warn(f'cannot run gz topic: {exc}')
            return
        if done.returncode != 0:
            self._warn(f'gz topic exited {done.returncode}: {done.stderr.strip()}')
        else:
            self._last_warning = None

    def _warn(self, text: str) -> None:
        """Log a failure once instead of at the publish rate."""
        if text != self._last_warning:
            self._last_warning = text
            self.get_logger().warning(text)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FollowCamera()
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
