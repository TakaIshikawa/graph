"""Detect reliability requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("availability", "high", (r"\bavailability\b", r"\bhighly\s+available\b")),
    ("disaster_recovery", "high", (r"\bdisaster\s+recovery\b", r"\bdr\s+plan\b")),
    ("failover", "high", (r"\bfailover\b", r"\bfail\s+over\b")),
    ("redundancy", "medium", (r"\bredundan(?:cy|t)\b", r"\breplication\b")),
    ("retry", "medium", (r"\bretr(?:y|ies)\b", r"\bretry\s+policy\b", r"\bbackoff\b")),
    ("rpo", "high", (r"\brpo\b", r"\brecovery\s+point\s+objective\b")),
    ("rto", "high", (r"\brto\b", r"\brecovery\s+time\s+objective\b")),
    ("uptime", "high", (r"\buptime\b", r"\b\d+(?:\.\d+)?%\s+uptime\b")),
)


def detect_query_reliability_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    if not text:
        return []

    rows: list[dict[str, Any]] = []
    for category, severity, patterns in _CATEGORIES:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
