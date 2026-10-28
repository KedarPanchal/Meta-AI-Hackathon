"""
Where commands go. Kept free of rclpy so the whole brain stack is testable without ROS.

:class:`fly_drone.bridge.TwistBridge` is the real one; the ROS node injects it into the
controller at startup. Everything here is a stand-in.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schemas import Command


@runtime_checkable
class Bridge(Protocol):
    """Anything that can carry a velocity command to a vehicle."""

    name: str

    def send(self, cmd: Command) -> None: ...

    def stop(self) -> None: ...


class NullBridge:
    """Accepts commands and drops them. The default, so nothing moves by accident."""

    name = 'null'

    def __init__(self) -> None:
        self.sent = 0
        self.last: Command | None = None

    def send(self, cmd: Command) -> None:
        self.sent += 1
        self.last = cmd

    def stop(self) -> None:
        self.send(Command(state='HOLD', reason='stop'))


class RecordingBridge(NullBridge):
    """Keeps every command, for tests and for replaying a run."""

    name = 'recording'

    def __init__(self) -> None:
        super().__init__()
        self.history: list[Command] = []

    def send(self, cmd: Command) -> None:
        super().send(cmd)
        self.history.append(cmd)
