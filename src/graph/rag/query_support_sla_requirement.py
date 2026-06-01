"""Detect support SLA requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("escalation", "high", (r"\bescalation\b", r"\bescalate\b")),
    ("incident_support", "high", (r"\bincident\s+support\b", r"\bproduction\s+support\b")),
    ("response_time", "high", (r"\bresponse\s+time\b", r"\btime\s+to\s+respond\b")),
    ("sla", "high", (r"\bsla\b", r"\bservice[-\s]?level\s+agreement\b")),
    ("support_window", "medium", (r"\bsupport\s+window\b", r"\bsupport\s+hours\b")),
    ("uptime_commitment", "high", (r"\buptime\s+commitment\b", r"\bavailability\s+commitment\b")),
)


def detect_query_support_sla_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
