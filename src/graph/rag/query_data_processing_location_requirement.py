"""Detect data processing and storage location requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_RESIDENCY_ONLY_RE = re.compile(r"^\s*(?:data\s+)?residency(?:\s+requirements?)?\s*$", re.I)
_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("cross_border_transfer", "high", (r"\bcross[-\s]?border\s+(?:data\s+)?transfers?\b", r"\btransfer\s+data\s+across\s+borders\b")),
    ("customer_selectable_region", "high", (r"\bcustomer[-\s]?selectable\s+region\b", r"\bcustomers?\s+can\s+choose\s+(?:the\s+)?region\b")),
    (
        "processing_region",
        "high",
        (r"\bwhere\s+(?:is\s+)?data\s+(?:is\s+)?processed\b", r"\bprocessing\s+region\b", r"\bdata\s+processed\s+in\b"),
    ),
    ("regional_failover_location", "medium", (r"\bregional\s+failover\b", r"\bfailover\s+region\b")),
    ("storage_region", "high", (r"\bwhere\s+(?:is\s+)?data\s+stored\b", r"\bstorage\s+region\b", r"\bdata\s+stored\s+in\b")),
    ("subprocessor_location", "medium", (r"\bsubprocessors?\s+by\s+location\b", r"\bsubprocessor\s+locations?\b")),
)


def detect_query_data_processing_location_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    if _RESIDENCY_ONLY_RE.match(text):
        return []
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
