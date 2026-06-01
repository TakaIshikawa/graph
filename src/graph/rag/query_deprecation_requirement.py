"""Detect deprecation planning requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("deprecation", "high", (r"\bdeprecat(?:e|ed|ion)\b",)),
    ("end_of_life", "high", (r"\bend[-\s]?of[-\s]?life\b", r"\beol\b")),
    ("removal_timeline", "high", (r"\bremoval\s+timeline\b", r"\bretirement\s+timeline\b")),
    ("replacement_path", "medium", (r"\breplacement\s+path\b", r"\bmigration\s+path\b")),
    ("sunset", "high", (r"\bsunset(?:ting)?\b",)),
)


def detect_query_deprecation_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
