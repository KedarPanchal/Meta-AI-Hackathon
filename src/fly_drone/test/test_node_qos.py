"""The pilot node's subscription QoS: reliable for images and lidar.

Same contract as mcp-server/tests/test_link_qos.py, other end of the same topics:
the ros-gz bridge publishes RELIABLE, and a best-effort subscription to a large
Image or PointCloud2 loses essentially every message on the macOS loopback, which
blinds the reflexes (forward camera, 3-D lidar) on this platform. Odometry and
camera info are small and high-rate, so they stay best-effort on purpose.

No live ROS: rclpy, rclpy.node and the message packages are faked in
sys.modules before fly_drone.node is imported.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest


class _SensorDataQoS:
    """Stands in for QoSPresetProfiles.SENSOR_DATA.value; identity-compared."""

    name = 'sensor-data-best-effort'


SENSOR_DATA = _SensorDataQoS()


class _FakePresets:
    SENSOR_DATA = SimpleNamespace(value=SENSOR_DATA)


class _FakeQoSProfile:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeReliability:
    RELIABLE = 'reliable'


class _FakeDurability:
    TRANSIENT_LOCAL = 'transient-local'


class _FakePub:
    def __init__(self, topic):
        self.topic = topic
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


class _FakeNodeBase:
    def __init__(self, name):
        self._name = name
        self.subscriptions = []
        self.publishers = []
        self.timers = []
        self._params = {}

    def declare_parameter(self, name, default):
        self._params[name] = default

    def get_parameter(self, name):
        return SimpleNamespace(value=self._params[name])

    def create_publisher(self, msg_type, topic, qos):
        self.publishers.append((msg_type, topic, qos))
        return _FakePub(topic)

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions.append((msg_type, topic, qos))
        return ('sub', topic)

    def create_timer(self, period, callback):
        self.timers.append((period, callback))
        return ('timer', period)

    def get_logger(self):
        return SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0))

    def destroy_node(self):
        pass


def _msg_class(name, **fields):
    def __init__(self):
        for key, factory in fields.items():
            setattr(self, key, factory())

    return type(name, (), {'__init__': __init__})


_Twist = _msg_class(
    'Twist',
    linear=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
    angular=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
)


def _with_data(value):
    return lambda self: setattr(self, 'data', value)


def _install_fakes():
    saved = {}
    rclpy = types.ModuleType('rclpy')
    rclpy_node = types.ModuleType('rclpy.node')
    rclpy_node.Node = _FakeNodeBase
    rclpy_param = types.ModuleType('rclpy.parameter')
    rclpy_param.Parameter = type('Parameter', (), {})

    class _QosModule(types.ModuleType):
        QoSPresetProfiles = _FakePresets
        QoSProfile = _FakeQoSProfile
        ReliabilityPolicy = _FakeReliability
        DurabilityPolicy = _FakeDurability

    rclpy_qos = _QosModule('rclpy.qos')
    mods = {'rclpy': rclpy, 'rclpy.node': rclpy_node,
            'rclpy.parameter': rclpy_param, 'rclpy.qos': rclpy_qos}

    def _pkg(name, **classes):
        top, _, _ = name.partition('.')
        mods.setdefault(top, types.ModuleType(top))
        mod = types.ModuleType(name)
        for cls_name, cls in classes.items():
            setattr(mod, cls_name, cls)
        mods[name] = mod

    _pkg('geometry_msgs.msg', Twist=_Twist, TwistStamped=_Twist)
    _pkg('nav_msgs.msg', Odometry=type('Odometry', (), {}))
    _pkg('sensor_msgs.msg', Image=type('Image', (), {}),
         CameraInfo=type('CameraInfo', (), {}), PointCloud2=type('PointCloud2', (), {}))
    _pkg('std_msgs.msg', String=type('String', (), {'__init__': _with_data('')}),
         Bool=type('Bool', (), {'__init__': _with_data(False)}))
    for name, mod in mods.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    return saved


@pytest.fixture
def fake_ros():
    saved = _install_fakes()
    try:
        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


@pytest.fixture
def pilot(fake_ros, monkeypatch):
    monkeypatch.setenv('FLYDRONE_DODGE_ADVISOR', '0')
    from fly_drone.node import BrainDroneNode

    return BrainDroneNode()


def _qos_by_topic(node):
    return {topic: qos for _cls, topic, qos in node.subscriptions}


def test_image_and_lidar_use_reliable_qos(pilot):
    qos = _qos_by_topic(pilot)
    assert qos[pilot.cfg.topic_image] == 10
    assert qos[pilot.cfg.topic_lidar] == 10


def test_odom_and_camera_info_stay_best_effort(pilot):
    qos = _qos_by_topic(pilot)
    assert qos[pilot.cfg.topic_odom] is SENSOR_DATA
    assert qos[pilot.cfg.topic_camera_info] is SENSOR_DATA
    assert qos[pilot.cfg.topic_goal] == 10
