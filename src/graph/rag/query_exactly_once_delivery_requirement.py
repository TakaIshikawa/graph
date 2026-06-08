"""Detect delivery semantics requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("exactly_once", (r"\bexactly[-\s]?once\b",)),
    ("at_least_once", (r"\bat[-\s]?least[-\s]?once\b",)),
    ("at_most_once", (r"\bat[-\s]?most[-\s]?once\b",)),
    ("deduplication_window", (r"\bdeduplication\s+windows?\b", r"\bdedupe\s+windows?\b")),
    ("delivery_semantics", (r"\bdelivery\s+semantics\b", r"\bmessage\s+delivery\s+guarantees?\b")),
)


def detect_query_exactly_once_delivery_requirement(query: str) -> dict[str, Any]:
    """Return delivery semantics requirement categories mentioned by a query."""
    requirements = _requirements(query)
    return {"has_exactly_once_delivery_requirement": bool(requirements), "requirements": requirements}


def _requirements(query: str) -> list[dict[str, str]]:
    text = " ".join(str(query or "").split())
    rows: list[dict[str, str]] = []
    for category, patterns in _CATEGORIES:
        match = _first_match(patterns, text)
        if match:
            rows.append({"category": category, "matched_text": match.group(0)})
    return sorted(rows, key=lambda row: row["category"])


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None
