"""Detect browser compatibility requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_BROWSERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("chrome", (r"\bchrome\b", r"\bchromium\b")),
    ("edge", (r"\bedge\b", r"\bmicrosoft\s+edge\b")),
    ("firefox", (r"\bfirefox\b",)),
    ("mobile_browser", (r"\bmobile\s+browsers?\b", r"\bios\s+safari\b", r"\bandroid\s+browser\b")),
    ("safari", (r"\bsafari\b",)),
)
_LEGACY = (r"\blegacy\s+browsers?\b", r"\bie\s*11\b", r"\binternet\s+explorer\b", r"\b(?:chrome|safari|firefox|edge)\s*(?:<=|<|before|older than)\s*\d+")


def detect_query_browser_compatibility_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    browsers = [name for name, patterns in _BROWSERS if any(re.search(pattern, text, re.I) for pattern in patterns)]
    legacy = any(re.search(pattern, text, re.I) for pattern in _LEGACY)
    return {
        "requires_browser_compatibility": bool(browsers or legacy),
        "browsers": browsers,
        "legacy_support_required": legacy,
    }
