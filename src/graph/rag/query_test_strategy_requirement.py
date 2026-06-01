"""Detect test strategy requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("acceptance_tests", "high", (r"\bacceptance\s+tests?\b", r"\bacceptance\s+criteria\b")),
    ("qa", "medium", (r"\bqa\b", r"\bquality\s+assurance\b")),
    ("regression_tests", "high", (r"\bregression\s+tests?\b", r"\bregression\s+suite\b")),
    ("test_plan", "high", (r"\btest\s+plans?\b", r"\btesting\s+strategy\b")),
    ("validation_criteria", "high", (r"\bvalidation\s+criteria\b", r"\bvalidation\s+plan\b")),
    ("verification_steps", "medium", (r"\bverification\s+steps?\b", r"\bverify\s+steps?\b")),
)


def detect_query_test_strategy_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
