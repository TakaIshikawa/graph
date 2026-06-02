"""Detect right-to-deletion requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("account_deletion", "high", (r"\baccount\s+deletion\b", r"\bdelete\s+(?:a\s+)?(?:user\s+)?account\b")),
    ("customer_data_deletion", "high", (r"\bcustomer\s+data\s+deletion\b", r"\bdelete\s+customer\s+data\b")),
    (
        "deletion_sla",
        "high",
        (r"\bdeletion\s+sla\b", r"\bdelete\s+(?:customer\s+data|data|records|accounts?)\s+within\b", r"\bdeletion\s+within\s+\d+"),
    ),
    ("purge_confirmation", "high", (r"\bpurge\s+confirmation\b", r"\bconfirm(?:ation)?\s+(?:of\s+)?(?:data\s+)?purge\b")),
    ("right_to_deletion", "high", (r"\bright\s+to\s+deletion\b", r"\bright\s+to\s+delete\b")),
    ("right_to_erasure", "high", (r"\bright\s+to\s+erasure\b", r"\berasure\s+request\b")),
)


def detect_query_right_to_deletion_requirements(query: str) -> list[dict[str, Any]]:
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
