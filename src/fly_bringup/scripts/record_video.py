#!/usr/bin/env python3
"""
Record the Gazebo flight to ``dist/<world>_<timestamp>.mp4``.

The launch starts this node next to ``ros_gz_bridge``. It subscribes to the
X3's ``camera_chase`` image - the server-side camera framed like the GUI's
follow camera - and streams the raw frames into ffmpeg. When the launch shuts
down, the encoder is closed and the finished file is moved into ``dist``. The
video is written under a partial name until ffmpeg has written its footer, so a
file in ``dist`` is always complete.

Parameters
----------
image_topic : str
    Bridged ``sensor_msgs/Image`` topic to record.
output_dir : str
    Directory for the finished video; created if missing.
fps : float
    Frame rate to encode. Match the sensor's ``<update_rate>``.
world : str
    World name used in the file name.
queue_size : int
    Frames buffered between the ROS callback and the encoder. When the encoder
    falls behind, the oldest buffered frame is dropped rather than blocking the
    executor or growing without bound.

"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

# sensor_msgs encoding -> (channels, ffmpeg pixel format)
PIXEL_FORMATS = {
    'rgb8': (3, 'rgb24'),
    'bgr8': (3, 'bgr24'),
    'rgba8': (4, 'rgba'),
    'bgra8': (4, 'bgra'),
    'mono8': (1, 'gray'),
}


class VideoRecorder(Node):
    """Feeds the chase-camera stream to ffmpeg and finalises it on shutdown."""

    def __init__(self) -> None:
        super().__init__('gazebo_video_recorder')
        self.declare_parameter('image_topic', '/drone/camera_chase_image')
        self.declare_parameter('output_dir', 'dist')
        self.declare_parameter('fps', 20.0)
        self.declare_parameter('world', 'gazebo')
        self.declare_parameter('queue_size', 8)

        self.image_topic = str(self.get_parameter('image_topic').value).strip()
        self.output_dir = os.path.abspath(
            os.path.expanduser(str(self.get_parameter('output_dir').value))
        )
        self.fps = max(float(self.get_parameter('fps').value), 1.0)
        world = str(self.get_parameter('world').value).strip() or 'gazebo'
        safe_world = ''.join(c if c.isalnum() or c in '-_' else '_' for c in world)
        stamp = time.strftime('%Y%m%d-%H%M%S')
        self.final_path = os.path.join(self.output_dir, f'{safe_world}_{stamp}.mp4')
        self.temp_path = os.path.join(self.output_dir, f'.{safe_world}_{stamp}.partial.mp4')
        self.ffmpeg = shutil.which('ffmpeg')

        os.makedirs(self.output_dir, exist_ok=True)
        if self.ffmpeg is None:
            self.get_logger().error(
                'ffmpeg is not on PATH; start this node inside the pixi environment, '
                'no video will be recorded'
            )

        self._queue: queue.Queue = queue.Queue(
            maxsize=max(int(self.get_parameter('queue_size').value), 1)
        )
        self._stop = threading.Event()
        self._stopped = False
        self._frames = 0
        self._dropped = 0
        self._encoder_ok = False
        self._encoder_error = ''
        self._warned_encoding = False
        self._thread = threading.Thread(target=self._drain, name='video-encoder', daemon=True)
        self._thread.start()

        from sensor_msgs.msg import Image

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(Image, self.image_topic, self._on_image, qos)
        self.create_timer(0.25, self._wake)
        self.get_logger().info(
            f'recording {self.image_topic} at {self.fps:g} fps; '
            f'the video will be written to {self.final_path} on shutdown'
        )

    # --- subscription --------------------------------------------------------

    def _wake(self) -> None:
        """No-op timer: keeps spin responsive to SIGINT while frames are idle."""

    def _on_image(self, msg) -> None:
        """Queue the newest frame; the encoder thread does the blocking work."""
        if self._stop.is_set():
            return
        encoding = str(msg.encoding).lower()
        if encoding not in PIXEL_FORMATS or msg.width <= 0 or msg.height <= 0:
            if not self._warned_encoding:
                self._warned_encoding = True
                self.get_logger().warning(f'cannot record image encoding {msg.encoding!r}')
            return

        channels = PIXEL_FORMATS[encoding][0]
        row_bytes = msg.width * channels
        data = bytes(msg.data)
        if msg.step != row_bytes:
            # padded rows: ffmpeg expects a packed frame
            data = b''.join(
                data[y * msg.step:y * msg.step + row_bytes] for y in range(msg.height)
            )
        if len(data) < row_bytes * msg.height:
            self.get_logger().warning(
                f'short image frame: {len(data)} bytes, expected {row_bytes * msg.height}',
                once=True,
            )
            return

        item = (data, msg.width, msg.height, encoding)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass
            self._dropped += 1

    # --- encoder thread ------------------------------------------------------

    def _drain(self) -> None:
        """Write queued frames to ffmpeg until stopped and drained."""
        proc = None
        try:
            while not (self._stop.is_set() and self._queue.empty()):
                try:
                    data, width, height, encoding = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if proc is None:
                    proc = self._start_encoder(width, height, encoding)
                    if proc is None:
                        continue
                try:
                    proc.stdin.write(data)
                except (BrokenPipeError, OSError, ValueError) as exc:
                    self._encoder_error = f'{type(exc).__name__}: {exc}'
                    break
                self._frames += 1
        finally:
            if proc is not None:
                self._stop_encoder(proc)

    def _start_encoder(self, width: int, height: int, encoding: str):
        if self.ffmpeg is None:
            return None
        if width % 2 or height % 2:
            self._encoder_error = f'odd frame size {width}x{height}; H.264 needs even dimensions'
            self.get_logger().error(self._encoder_error)
            return None

        pix_fmt = PIXEL_FORMATS[encoding][1]
        command = [
            self.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', pix_fmt,
            '-s', f'{width}x{height}', '-r', f'{self.fps:g}', '-i', 'pipe:0',
            '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            self.temp_path,
        ]
        try:
            # Ctrl-C reaches every process in the launch's group, ffmpeg included.
            # If ffmpeg sees SIGINT it writes its trailer and exits 255, so the
            # recording looks failed; isolate it and stop it by closing stdin.
            proc = subprocess.Popen(
                command, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            self._encoder_error = f'cannot start ffmpeg: {exc}'
            self.get_logger().error(self._encoder_error)
            return None
        self.get_logger().info(f'encoder started for {width}x{height} {encoding}')
        return proc

    def _stop_encoder(self, proc) -> None:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
            self._encoder_error = self._encoder_error or 'ffmpeg timed out while closing'
        stderr = b''
        if proc.stderr is not None:
            try:
                stderr = proc.stderr.read()
            except OSError:
                pass
        if code != 0 or self._encoder_error:
            detail = self._encoder_error or stderr.decode('utf-8', 'replace')[-800:]
            self._say('error', f'ffmpeg exited {code}: {detail}')
        else:
            self._encoder_ok = True

    # --- shutdown ------------------------------------------------------------

    def _say(self, level: str, text: str) -> None:
        """
        Log through ROS while the context is alive, else straight to stdout.

        A SIGINT from the launch shuts the context down before ``spin`` returns,
        so the shutdown messages can no longer reach rosout.
        """
        if rclpy.ok():
            getattr(self.get_logger(), level)(text)
        else:
            print(f'[{level.upper()}] {text}', flush=True)

    def shutdown(self) -> None:
        """Stop the encoder and move the complete video into ``dist``."""
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        self._thread.join(timeout=45)

        if self._thread.is_alive():
            self._say(
                'error', f'encoder did not finish; partial recording left at {self.temp_path}'
            )
            return
        if not self._frames:
            self._say('warning', 'no camera frames arrived; no video was written')
            try:
                os.unlink(self.temp_path)
            except OSError:
                pass
            return
        if not self._encoder_ok or not os.path.exists(self.temp_path):
            self._say('error', f'recording failed; partial video left at {self.temp_path}')
            return
        try:
            os.replace(self.temp_path, self.final_path)
        except OSError as exc:
            self._say('error', f'cannot move video into {self.output_dir}: {exc}')
            return
        self._say(
            'info', f'video saved: {self.final_path} '
            f'({self._frames} frames, {self._dropped} dropped)'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VideoRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
