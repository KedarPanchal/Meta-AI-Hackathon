"""The raw Muse-output log: the payload shape and the no-ROS failure mode."""
from __future__ import annotations

import json

from mission.muse_log import MuseRawLog


def test_record_keeps_the_whole_answer_and_is_json_ready() -> None:
    raw = '{"person_visible": true, "description": "' + 'x' * 1200 + '"}'
    rec = MuseRawLog.record('detector', 'muse-spark-1.3', raw, ok=True,
                            latency_ms=812.44, height_m=15.0)
    assert rec['role'] == 'detector' and rec['model'] == 'muse-spark-1.3'
    assert rec['ok'] is True and rec['error'] is None
    assert rec['raw'] == raw                 # complete, not Detection.raw's 500-char copy
    assert rec['latency_ms'] == 812.4
    assert rec['context'] == {'height_m': 15.0}
    json.dumps(rec)


def test_disabled_log_never_touches_ros() -> None:
    log = MuseRawLog('/mission/muse_raw', enabled=False)
    log.publish('navigator', 'm', 'answer', ok=True)
    assert log.sent == 0 and log.dropped == 1 and log.describe()['publisher'] is False


def test_attached_publisher_avoids_a_second_ros_node() -> None:
    records = []
    log = MuseRawLog('/mission/muse_raw', publisher=records.append)
    log.ensure()
    log.publish('navigator', 'm', 'answer', ok=True, latency_ms=2.0, height_m=15.0)
    assert log.sent == 1 and log.dropped == 0
    assert log.describe()['publisher'] is True
    assert log.describe()['external'] is True
    assert records[0]['role'] == 'navigator' and records[0]['raw'] == 'answer'
    assert records[0]['context'] == {'height_m': 15.0}
    log.close()
    assert log.describe()['publisher'] is False


def test_after_close_a_late_answer_is_dropped() -> None:
    """A Muse call can outlive the abort join; it must not recreate a ROS node."""
    records = []
    log = MuseRawLog('/mission/muse_raw', publisher=records.append)
    log.close()
    log.ensure()
    log.publish('detector', 'm', 'late answer', ok=True)
    assert log.sent == 0 and log.dropped == 1
    assert records == []
    assert log.describe()['publisher'] is False
    assert log.describe()['closed'] is True
