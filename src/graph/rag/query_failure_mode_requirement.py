"""Detect failure-mode requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("risk", "high", (r"\brisks?\b", r"\bwhat\s+could\s+go\s+wrong\b")),
    ("failure_mode", "high", (r"\bfailure\s+modes?\b", r"\bfailures?\b")),
    ("pitfall", "medium", (r"\bpitfalls?\b", r"\bgotchas?\b")),
    ("edge_case", "medium", (r"\bedge\s+cases?\b", r"\bcorner\s+cases?\b")),
    ("rollback", "high", (r"\brollback\s+plans?\b", r"\broll\s+back\b", r"\brecovery\s+plans?\b")),
)


def detect_query_failure_mode_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
