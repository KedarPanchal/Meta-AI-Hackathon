"""Test environment: the in-process sim link, the colour detector, the mock brain, fast time.

Set before anything imports ``fly_brain`` or ``mission`` so the settings singletons see it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault('BRAIN_BACKEND', 'mock')
os.environ.setdefault('MISSION_LINK', 'sim')
os.environ.setdefault('PERSON_DETECTOR', 'color')
os.environ.setdefault('SIM_SPEEDUP', '40')
os.environ.setdefault('SIM_DT', '0.05')
os.environ.setdefault('MISSION_POLL_S', '0.01')
os.environ.setdefault('MISSION_LEG_TIMEOUT_MIN_S', '30')
