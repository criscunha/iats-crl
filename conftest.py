"""Pytest configuration: make the repository root importable.

Ensures the ``experiments`` scripts can be imported from tests, alongside the
installed ``iats`` package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
