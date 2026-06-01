"""Detect data residency requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("cross_border_transfer", "high", (r"\bcross[-\s]?border\s+transfers?\b", r"\bdata\s+transfer\s+limits?\b")),
    ("data_residency", "high", (r"\bdata\s+residency\b", r"\bdata\s+locali[sz]ation\b")),
    ("eu_only", "high", (r"\beu[-\s]?only\b", r"\bwithin\s+the\s+eu\b")),
    ("jurisdictional_storage", "high", (r"\bjurisdictional\s+storage\b", r"\bstored\s+in\s+jurisdiction\b")),
    ("regional_hosting", "medium", (r"\bregional\s+hosting\b", r"\bhosted\s+in\s+(?:region|country)\b")),
    ("sovereign_cloud", "high", (r"\bsovereign\s+cloud\b",)),
    ("us_only", "high", (r"\bus[-\s]?only\b", r"\bwithin\s+the\s+us\b")),
)


def detect_query_data_residency_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
