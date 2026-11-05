"""The ROS link's subscription QoS: reliable for images, best-effort for odometry.

The ros-gz bridge publishes every topic RELIABLE / KEEP_LAST(10). A best-effort
(SENSOR_DATA) subscription to a small topic such as odometry works, but a 921 kB
camera frame over best-effort is lost all but every time on the macOS loopback:
no retransmit means any dropped fragment drops the frame. The mission layer then
never sees a downward frame, the detector never runs, and nothing is published
on /mission/muse_raw. Images must subscribe reliably; odometry stays best-effort
on purpose (high rate, tiny, newest-only).

No live ROS here: rclpy and the message packages are faked in sys.modules, so
this runs in the fast suite and in a sandbox.
"""
from __future__ import annotations

import sys
import types

import pytest


class _SensorDataQoS:
    """Stands in for QoSPresetProfiles.SENSOR_DATA.value; identity-compared."""

    name = 'sensor-data-best-effort'


SENSOR_DATA = _SensorDataQoS()


class _FakePresets:
    SENSOR_DATA = types.SimpleNamespace(value=SENSOR_DATA)


class _FakeNode:
    def __init__(self):
        self.subscriptions = []
        self.publishers = []
        self.destroyed = False

    def create_publisher(self, msg_type, topic, qos):
        self.publishers.append((msg_type, topic, qos))
        return ('pub', topic)

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions.append((msg_type, topic, qos))
        return ('sub', topic)

    def destroy_node(self):
        self.destroyed = True


class _FakeExecutor:
    def __init__(self):
        self.nodes = []
        self.shutdown_called = False

    def add_node(self, node):
        self.nodes.append(node)

    def spin(self):
        pass

    def shutdown(self, timeout_sec=None):
        self.shutdown_called = True
        return True

    def remove_node(self, node):
        self.nodes.remove(node)


@pytest.fixture
def fake_ros():
    """Fake rclpy + message modules; RosLink imports them lazily in __init__."""
    saved = {}
    node_holder = {}

    rclpy = types.ModuleType('rclpy')
    rclpy.ok = lambda: True
    rclpy.init = lambda args=None: None
    rclpy.shutdown = lambda: None

    def _create_node(name):
        node = _FakeNode()
        node_holder['node'] = node
        return node

    rclpy.create_node = _create_node

    executors = types.ModuleType('rclpy.executors')
    executors.SingleThreadedExecutor = _FakeExecutor
    qos = types.ModuleType('rclpy.qos')
    qos.QoSPresetProfiles = _FakePresets
    nav_msgs = types.ModuleType('nav_msgs')
    nav_msg = types.ModuleType('nav_msgs.msg')

    class Odometry:
        pass

    nav_msg.Odometry = Odometry
    sensor_msgs = types.ModuleType('sensor_msgs')
    sensor_msg = types.ModuleType('sensor_msgs.msg')

    class Image:
        pass

    sensor_msg.Image = Image
    std_msgs = types.ModuleType('std_msgs')
    std_msg = types.ModuleType('std_msgs.msg')

    class String:
        def __init__(self):
            self.data = ''

    std_msg.String = String
    mods = {
        'rclpy': rclpy, 'rclpy.executors': executors, 'rclpy.qos': qos,
        'nav_msgs': nav_msgs, 'nav_msgs.msg': nav_msg,
        'sensor_msgs': sensor_msgs, 'sensor_msgs.msg': sensor_msg,
        'std_msgs': std_msgs, 'std_msgs.msg': std_msg,
    }
    for name, mod in mods.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    try:
        yield node_holder
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _qos_by_topic(node):
    return {topic: qos for _cls, topic, qos in node.subscriptions}


def test_downward_image_uses_reliable_qos(fake_ros):
    from mission.config import MissionSettings
    from mission.link import RosLink

    cfg = MissionSettings()
    cfg.link = 'ros'
    link = RosLink(cfg)
    try:
        qos = _qos_by_topic(fake_ros['node'])
        assert qos[cfg.brain.topic_image_down] == 10
    finally:
        link.close()


def test_odometry_stays_best_effort_and_status_reliable(fake_ros):
    from mission.config import MissionSettings
    from mission.link import RosLink

    cfg = MissionSettings()
    cfg.link = 'ros'
    link = RosLink(cfg)
    try:
        qos = _qos_by_topic(fake_ros['node'])
        assert qos[cfg.brain.topic_odom] is SENSOR_DATA
        assert qos[cfg.brain.topic_goal_status] == 10
    finally:
        link.close()


def test_close_shuts_down_executor_and_destroys_node(fake_ros):
    from mission.config import MissionSettings
    from mission.link import RosLink

    cfg = MissionSettings()
    cfg.link = 'ros'
    link = RosLink(cfg)
    link.close()
    assert fake_ros['node'].destroyed
    assert link._exec.shutdown_called
