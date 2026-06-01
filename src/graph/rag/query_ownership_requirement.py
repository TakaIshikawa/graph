"""Detect ownership and responsibility requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("accountability", "high", (r"\baccountable\b", r"\baccountability\b")),
    ("decision_maker", "medium", (r"\bdecision[-\s]?makers?\b", r"\bwho\s+decides\b")),
    ("escalation_path", "high", (r"\bescalation\s+paths?\b", r"\bescalate\s+to\b")),
    ("owner", "high", (r"\bowners?\b", r"\bownership\b")),
    ("raci", "medium", (r"\braci\b",)),
    ("responsibility_matrix", "medium", (r"\bresponsibility\s+matrix\b", r"\bresponsible\s+part(?:y|ies)\b")),
)


def detect_query_ownership_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
