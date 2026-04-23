"""Test configuration for local source and dependency resolution."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

venv_site_packages = sorted(ROOT.glob(".venv/lib/python*/site-packages"))
for path in reversed(venv_site_packages):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
