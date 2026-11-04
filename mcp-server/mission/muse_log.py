"""
Every raw Muse answer, on a ROS 2 topic.

Muse is asked three different questions -- is a person there, where to go next, which way to
dodge -- from two processes: the mission layer for the detector and navigator, and
``fly_brain_pilot`` for the dodge advisor. Debugging a vision model's decisions means seeing
what it actually said, not what the parser made of it, so each client hands the complete
answer text here. It is published as JSON on ``std_msgs/String`` at ``ROS_TOPIC_MUSE_RAW``
(default ``/mission/muse_raw``)::

    {"t": 1762300000.123, "role": "detector", "model": "muse-spark-1.3", "ok": true,
     "raw": "<the complete model text>", "error": null, "latency_ms": 812.4,
     "context": {"height_m": 15.0}}

A failure is published too, with ``ok`` false, ``error`` set and ``raw`` null when the model
never answered. The record is for debugging only; no consumer feeds it back into control.

This module is importable without ROS. rclpy and the publisher are created lazily on the
first answer, and publishing is best-effort: a missing ROS environment, a failed node or a
DDS error must never interrupt a flight. The dodge advisor lives in ``fly_drone`` and builds
the same record, because ``fly_drone`` must not import the mission package.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Protocol


class MuseRawSink(Protocol):
    """Where a Muse client sends the answer it just received."""

    def publish(self, role: str, model: str, raw: str | None, *, ok: bool,
                error: str | None = None, latency_ms: float | None = None,
                **context: object) -> None: ...


class MuseRawLog:
    """Best-effort JSON publisher for raw Muse answers; creates its ROS node on demand."""

    def __init__(self, topic: str = '/mission/muse_raw', node_name: str = 'muse_raw_log',
                 enabled: bool = True) -> None:
        self.topic = topic
        self.node_name = node_name
        self.enabled = enabled
        self.sent = 0
        self.dropped = 0
        self.last_error: str | None = None
        self._pub = None
        self._node = None
        self._String = None
        self._failed = False
        self._lock = threading.Lock()

    @staticmethod
    def record(role: str, model: str, raw: str | None, *, ok: bool,
               error: str | None = None, latency_ms: float | None = None,
               t: float | None = None, **context: object) -> dict:
        """The JSON payload. ``raw`` is the complete answer, never the truncated copy."""
        out = {
            't': round(time.time() if t is None else float(t), 3),
            'role': role,
            'model': model,
            'ok': bool(ok),
            'raw': raw,
            'error': error,
            'latency_ms': None if latency_ms is None else round(float(latency_ms), 1),
        }
        if context:
            out['context'] = context
        return out

    def ensure(self) -> None:
        """Create the publisher now, so the topic is visible before the first Muse answer."""
        with self._lock:
            self._ensure_publisher()

    def publish(self, role: str, model: str, raw: str | None, *, ok: bool,
                error: str | None = None, latency_ms: float | None = None,
                **context: object) -> None:
        if not self.enabled:
            self.dropped += 1
            return
        try:
            payload = self.record(role, model, raw, ok=ok, error=error,
                                  latency_ms=latency_ms, **context)
            with self._lock:
                self._ensure_publisher()
                if self._pub is None:
                    self.dropped += 1
                    return
                msg = self._String()
                msg.data = json.dumps(payload, default=str)
                self._pub.publish(msg)
                self.sent += 1
        except Exception as exc:                 # observability must not fail the flight
            self.last_error = f'{type(exc).__name__}: {exc}'
            self.dropped += 1

    def _ensure_publisher(self) -> None:
        if self._pub is not None or self._failed:
            return
        try:
            import rclpy
            from std_msgs.msg import String

            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = rclpy.create_node(self.node_name)
            self._pub = self._node.create_publisher(String, self.topic, 10)
            self._String = String
        except Exception as exc:
            # No ROS in this process is a normal case for the in-process sim and tests.
            self._failed = True
            self.last_error = f'{type(exc).__name__}: {exc}'

    def describe(self) -> dict:
        return {
            'topic': self.topic,
            'enabled': self.enabled,
            'sent': self.sent,
            'dropped': self.dropped,
            'publisher': self._pub is not None,
            'last_error': self.last_error,
        }

    def close(self) -> None:
        with self._lock:
            try:
                if self._node is not None:
                    self._node.destroy_node()
            except Exception:
                pass
            self._node = None
            self._pub = None
