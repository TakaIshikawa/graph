"""Detect subprocessor notification requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("objection_period", "high", (r"\bobjection\s+period\b", r"\b\d+\s+days?\s+to\s+object\b")),
    ("subprocessor_list", "high", (r"\bsubprocessors?\s+list\b", r"\blist\s+of\s+subprocessors?\b")),
    ("subprocessor_notice", "high", (r"\bsubprocessor\s+notice\b", r"\bsubprocessor\s+notification\b")),
    ("third_party_processor_update", "medium", (r"\bthird[-\s]?party\s+processor\s+updates?\b",)),
    ("vendor_change_notice", "high", (r"\bvendor\s+change\s+notice\b", r"\bnotice\s+of\s+(?:vendor|processor)\s+changes?\b")),
)


def detect_query_subprocessor_notification_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, requirement_strength, patterns in _CATEGORIES:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append(
                {
                    "matched_text": match.group(0),
                    "category": category,
                    "requirement_strength": requirement_strength,
                }
            )
    return sorted(rows, key=lambda row: row["category"])
