"""Detect audience expertise requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("beginner", "medium", (r"\bbeginners?\b", r"\bnew\s+to\s+this\b", r"\bintroductory\b")),
    ("developer", "medium", (r"\bdevelopers?\b", r"\bengineers?\b", r"\btechnical\s+implementers?\b")),
    ("executive", "medium", (r"\bexecutives?\b", r"\bleadership\b", r"\bc[-\s]?suite\b")),
    ("expert", "high", (r"\bexperts?\b", r"\badvanced\s+audience\b", r"\bdeep\s+technical\b")),
    ("non_technical", "high", (r"\bnon[-\s]?technical\b", r"\bplain\s+english\b", r"\blay(?:person|people)\b")),
    ("specialist", "high", (r"\bspecialists?\b", r"\bclinicians?\b", r"\bdata\s+scientists?\b")),
)


def detect_query_audience_expertise_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
