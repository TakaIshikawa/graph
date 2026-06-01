"""Detect human review requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("approval", "high", (r"\bhuman\s+approval\b", r"\bmanager\s+approval\b", r"\bapproval\s+before\b")),
    ("escalation", "high", (r"\bescalat(?:e|ion)\s+to\s+(?:a\s+)?human\b", r"\bhuman\s+escalation\b")),
    ("expert_validation", "high", (r"\bexpert\s+validation\b", r"\bexpert\s+review\b")),
    ("human_in_the_loop", "high", (r"\bhuman[-\s]?in[-\s]?the[-\s]?loop\b", r"\bhitl\b")),
    ("manual_signoff", "high", (r"\bmanual\s+sign[-\s]?off\b", r"\bhuman\s+sign[-\s]?off\b")),
)


def detect_query_human_review_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
