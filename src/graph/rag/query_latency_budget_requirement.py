"""Detect latency budget requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("latency", "high", (r"\blatency\b", r"\blatency\s+budget\b")),
    ("performance_budget", "medium", (r"\bperformance\s+budget\b",)),
    ("percentile_latency", "high", (r"\bp(?:95|99)\b", r"\b(?:95th|99th)\s+percentile\b")),
    ("realtime", "high", (r"\breal[-\s]?time\b", r"\bnear[-\s]?real[-\s]?time\b")),
    ("response_time", "high", (r"\bresponse\s+time\b", r"\btime\s+to\s+respond\b")),
    ("timeout", "high", (r"\btimeouts?\b", r"\btime\s+out\b")),
)


def detect_query_latency_budget_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
