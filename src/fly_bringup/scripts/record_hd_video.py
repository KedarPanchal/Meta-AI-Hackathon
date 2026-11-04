#!/usr/bin/env python3
"""
Record the HD chase camera to ``dist/<world>_<timestamp>_hd.mp4``.

This is the high-quality companion to ``record_video``. That node records
the X3's low-rate chase sensor (320x240 at 10 Hz); this one records the
dedicated ``camera_chase_hd`` sensor in ``models/x3_uav/model.sdf`` —
same chase framing (3 m behind, 1.5 m above the airframe), but
1920x1080 at 60 Hz. No other sensor, world, or physics setting changes,
and the recording works headless because nothing involves the GUI.

The encoding is done inside Gazebo by its ``CameraVideoRecorder`` system
plugin, which is attached to the HD sensor in the model and advertises a
gz service. This node drives that service with ``gz service``, the same
CLI pattern ``follow_camera`` uses:

* shortly after start, once the service appears: ``start`` with the
  final file name;
* on shutdown: ``stop``, then wait for the file to be written.

Shutdown ordering matters. The launch sends SIGINT to every process at
once, Gazebo included, so the stop call races Gazebo's own exit. The
server takes longer to tear down than this call takes, and the node's
``sigterm_timeout`` in ``sim_launch.py`` gives it the same long grace
as ``record_video``. If Gazebo is already gone (the operator closed
the window first), the stop call fails, a warning is logged, and the
sensor recording in ``dist`` is still there.

Note the HD sensor renders whenever Gazebo runs, whether or not a
recording is started: 1080p60 is a real render cost on top of the other
cameras. ``hd_recording`` only controls whether the encoder runs.

Also keep ``output_dir`` on the same filesystem as the launch's working
directory (the default ``dist`` is): the plugin finishes by renaming a
temporary file from the Gazebo server's cwd, and a rename across
filesystems (for example a tmpfs ``/tmp``) silently leaves no file.

Parameters
----------
service : str
    gz service of the sensor's CameraVideoRecorder plugin.
output_dir : str
    Directory for the finished video; created if missing.
world : str
    World name used in the file name.
start_timeout_s : float
    How long to wait for the recorder service to appear before giving
    up. Only the sensor video is recorded in that case; nothing fails.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

SERVICE = '/camera_chase_hd/record_video'
REQTYPE = 'gz.msgs.VideoRecord'
REPTYPE = 'gz.msgs.Boolean'


def service_request(start: bool, save_filename: str = '') -> str:
    """Build the protobuf-text request for the record_video service.

    The plugin encodes to a temporary file while recording and only
    moves it to ``save_filename`` when the *stop* request carries it;
    the name given at start is not remembered. Both calls therefore
    pass it.
    """
    escaped = save_filename.replace('\\', '\\\\').replace('"', '\\"')
    if start:
        return f'start: true, format: "mp4", save_filename: "{escaped}"'
    return f'stop: true, save_filename: "{escaped}"'


class HdVideoRecorder(Node):
    """Starts the HD camera's recorder on boot and stops it on shutdown."""

    def __init__(self) -> None:
        super().__init__('hd_video_recorder')
        self.declare_parameter('service', SERVICE)
        self.declare_parameter('output_dir', 'dist')
        self.declare_parameter('world', 'gazebo')
        self.declare_parameter('start_timeout_s', 90.0)

        self.service = str(self.get_parameter('service').value).strip() or SERVICE
        self.output_dir = os.path.abspath(
            os.path.expanduser(str(self.get_parameter('output_dir').value))
        )
        self.start_timeout_s = max(
            float(self.get_parameter('start_timeout_s').value), 1.0)
        world = str(self.get_parameter('world').value).strip() or 'gazebo'
        safe_world = ''.join(c if c.isalnum() or c in '-_' else '_' for c in world)
        stamp = time.strftime('%Y%m%d-%H%M%S')
        self.final_path = os.path.join(
            self.output_dir, f'{safe_world}_{stamp}_hd.mp4')
        self.gz = shutil.which('gz')

        os.makedirs(self.output_dir, exist_ok=True)
        self._recording = False
        self._stopped = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        if self.gz is None:
            self.get_logger().error(
                'gz is not on PATH; start this node inside the pixi environment, '
                'no HD video will be recorded')
            return

        self._thread = threading.Thread(
            target=self._start_when_ready, name='hd-video-starter', daemon=True)
        self._thread.start()
        self.get_logger().info(
            f'HD recording armed on {self.service}; '
            f'the video will be written to {self.final_path}')

    # --- gz service calls ----------------------------------------------------

    def _call(self, request: str, timeout_s: float) -> tuple[bool, str]:
        """Call the record_video service once. Returns (ok, combined output)."""
        command = [
            self.gz, 'service', '-s', self.service,
            '--reqtype', REQTYPE, '--reptype', REPTYPE,
            '--timeout', str(int(timeout_s * 1000)), '--req', request,
        ]
        try:
            done = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout_s + 10.0)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f'{type(exc).__name__}: {exc}'
        out = (done.stdout + done.stderr).strip()
        # A successful Boolean reply prints "true"; a timeout prints an error.
        return done.returncode == 0 and 'true' in out.lower(), out

    def _service_ready(self) -> bool:
        try:
            done = subprocess.run(
                [self.gz, 'service', '--list'],
                capture_output=True, text=True, timeout=10.0)
        except (OSError, subprocess.SubprocessError):
            return False
        return any(line.strip() == self.service
                   for line in done.stdout.splitlines())

    def _start_when_ready(self) -> None:
        deadline = time.monotonic() + self.start_timeout_s
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            if self._service_ready():
                ok, out = self._call(
                    service_request(True, self.final_path), timeout_s=15.0)
                if ok:
                    self._recording = True
                    self._say('info', f'HD recording started: {self.final_path}')
                else:
                    self._say(
                        'error',
                        f'HD recorder refused to start ({out[:200]}); '
                        'only the sensor video will be recorded')
                return
            self._stop_event.wait(1.0)
        if not self._stop_event.is_set():
            self._say(
                'warning',
                f'HD recorder service {self.service} never appeared within '
                f'{self.start_timeout_s:g} s (is the HD camera in the model?); '
                'only the sensor video will be recorded')

    # --- shutdown --------------------------------------------------------------

    def _say(self, level: str, text: str) -> None:
        """Log through ROS while the context is alive, else straight to stdout."""
        if rclpy.ok():
            getattr(self.get_logger(), level)(text)
        else:
            print(f'[{level.upper()}] {text}', flush=True)

    def shutdown(self) -> None:
        """Stop the HD recorder and wait for its file to land.

        The Makefile's found-exit path stops the recording *before*
        Ctrl-C'ing the launch, while Gazebo is still rendering, because
        the plugin only moves its temporary file on a render callback
        (see ``service_request``). That pre-stop makes this call the
        second one: the plugin is already idle, so it is harmless, and
        an already-saved file is reported without calling again.
        """
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        if not self._recording:
            return
        if os.path.exists(self.final_path) and os.path.getsize(self.final_path) > 0:
            self._say('info', f'HD video saved: {self.final_path} '
                              f'({os.path.getsize(self.final_path)} bytes)')
            return
        ok, out = self._call(
            service_request(False, self.final_path), timeout_s=20.0)
        # Even a failed call is not final: Gazebo may have processed the
        # stop on its way down. Always wait for the file before warning.
        deadline = time.monotonic() + 30.0
        last_size = -1
        while time.monotonic() < deadline:
            try:
                size = os.path.getsize(self.final_path)
            except OSError:
                size = -1
            if size > 0 and size == last_size:
                self._say('info', f'HD video saved: {self.final_path} ({size} bytes)')
                return
            last_size = size
            time.sleep(0.5)
        if not ok:
            self._say(
                'warning',
                f'HD recorder stop failed ({out[:200]}); Gazebo may '
                f'already have exited. Sensor video in {self.output_dir} '
                'is unaffected')
        else:
            self._say(
                'warning',
                f'HD video did not settle at {self.final_path}; '
                'check whether the file is still being written')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HdVideoRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        # RCLError is the same SIGINT wait-set race record_video handles:
        # a graceful shutdown whose finally finalises the recording.
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
