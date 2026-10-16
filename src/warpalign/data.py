"""Batch and the fish-oil loader (warpalign 2.x). Moved to alignbench.data and
alignbench.datasets.fish_oil.

Deprecated re-exports, kept so code written against warpalign 2.x still runs (same results:
alignbench's golden tests compare it with warpalign 2.0.0 bit for bit). Import from alignbench.
"""

from __future__ import annotations

from . import _compat
from ._compat import *  # noqa: F403
from ._compat import DATA

__all__ = DATA

_compat.deprecated('data')
