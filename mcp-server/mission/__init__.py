"""
The mission layer: *where* the drone should go, and what the camera says when it gets there.

It never commands velocity. It plans a coverage path (boustrophedon decomposition), hands the
brain one goal at a time, and runs Meta's Muse Spark over the downward camera to find a person.
Reflexes and obstacle avoidance stay in the brain (``fly_brain``), which is why the only
message between the two is a world position.
"""

import sys as _sys
from pathlib import Path as _Path

try:
    import fly_brain  # noqa: F401
except ImportError:                                   # unbuilt workspace: use the source tree
    _src = _Path(__file__).resolve().parents[2] / 'src' / 'fly_brain'
    if str(_src) not in _sys.path:
        _sys.path.insert(0, str(_src))
    import fly_brain  # noqa: F401,E402
