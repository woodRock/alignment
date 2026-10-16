"""Calibration anchors (warpalign 2.x). Moved to alignbench.anchors (extract_anchors =
match_pooled_arrays).

Deprecated re-exports, kept so code written against warpalign 2.x still runs (same results:
alignbench's golden tests compare it with warpalign 2.0.0 bit for bit). Import from alignbench.
"""

from __future__ import annotations

from . import _compat
from ._compat import *  # noqa: F403
from ._compat import ANCHORS

__all__ = ANCHORS

_compat.deprecated('anchors')
