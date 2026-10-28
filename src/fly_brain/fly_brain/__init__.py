"""
fly_brain: the connectome brain and its decoders, with no ROS dependency.

The ROS side lives in :mod:`fly_drone`; everything here runs and is testable without rclpy,
which is what lets the reflex be calibrated offline.
"""
from .circuit import Circuit
from .config import reload_settings, Settings, settings
from .controller import Controller, controller, reset_controller
from .schemas import (
    Box,
    BrainOut,
    BrainStatus,
    Command,
    DroneState,
    Percept,
    PursuitParams,
    ReflexEvent,
    ReflexParams,
)

__all__ = [
    'Box', 'BrainOut', 'BrainStatus', 'Circuit', 'Command', 'Controller', 'DroneState',
    'Percept', 'PursuitParams', 'ReflexEvent', 'ReflexParams', 'Settings', 'controller',
    'reload_settings', 'reset_controller', 'settings',
]
