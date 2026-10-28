"""Make ``fly_brain`` importable whether or not the workspace has been built.

After ``colcon build`` plus sourcing the install, ``fly_brain`` is on the path already and
this does nothing. Before that -- which is the normal case while developing the brain -- it
adds ``src/fly_brain`` directly. ``fly_drone`` is deliberately not added: it needs rclpy, and
the MCP server must run without ROS.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_fly_brain() -> str:
    try:
        import fly_brain  # noqa: F401

        return "installed"
    except ImportError:
        pass
    src = REPO_ROOT / "src" / "fly_brain"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import fly_brain  # noqa: F401

    return f"source ({src})"
