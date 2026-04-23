"""Test configuration for local source and dependency resolution."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VENV_SITE_PACKAGES = ROOT / ".venv" / "lib" / "python3.12" / "site-packages"

for path in (SRC, VENV_SITE_PACKAGES):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
