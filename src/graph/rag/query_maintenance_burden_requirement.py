"""Detect maintenance burden requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("handoff_effort", "medium", (r"\bhandoff\s+effort\b", r"\bknowledge\s+transfer\b", r"\btransfer\s+ownership\b")),
    ("long_term_support", "high", (r"\blong[-\s]?term\s+support\b", r"\bsupport\s+burden\b", r"\bsustain\s+over\s+time\b")),
    ("maintenance_burden", "high", (r"\bmaintenance\s+burden\b", r"\bmaintainability\b", r"\beasy\s+to\s+maintain\b")),
    ("ongoing_upkeep", "medium", (r"\bongoing\s+upkeep\b", r"\bupkeep\b", r"\bmaintenance\s+work\b")),
    ("operational_overhead", "high", (r"\boperational\s+overhead\b", r"\bops\s+overhead\b", r"\brun\s+cost\b")),
)


def detect_query_maintenance_burden_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
