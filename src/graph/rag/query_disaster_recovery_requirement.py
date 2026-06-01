"""Detect disaster recovery requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("backup_restore", "high", (r"\bbackup\s*/\s*restore\b", r"\bbackups?\b", r"\brestore\b")),
    ("business_continuity", "high", (r"\bbusiness\s+continuity\b", r"\bcontinuity\s+planning\b")),
    ("disaster_recovery", "high", (r"\bdisaster\s+recovery\b", r"\bdr\s+plan\b")),
    ("failover", "high", (r"\bfailover\b", r"\bfail\s+over\b")),
    ("recovery_plan", "high", (r"\brecovery\s+plans?\b", r"\brecovery\s+procedures?\b")),
    ("rpo", "high", (r"\brpo\b", r"\brecovery\s+point\s+objective\b")),
    ("rto", "high", (r"\brto\b", r"\brecovery\s+time\s+objective\b")),
)


def detect_query_disaster_recovery_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
