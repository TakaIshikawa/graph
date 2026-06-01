"""Detect cost constraint requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("budget", (r"\bbudget\b", r"\bwithin\s+budget\b", r"\bspending\s+limit\b")),
    ("license_cost", (r"\blicen[cs]e\s+costs?\b", r"\blicensing\s+fees?\b", r"\bseat\s+costs?\b")),
    ("pricing", (r"\bpricing\b", r"\bprice\s+comparison\b", r"\bcompare\s+prices?\b")),
    ("cost_cap", (r"\bcost\s+caps?\b", r"\bprice\s+caps?\b", r"\bnot\s+exceed\b", r"\bunder\s+\$?\d+(?:,\d{3})*(?:\.\d+)?")),
    ("tco", (r"\btotal\s+cost\s+of\s+ownership\b", r"\btco\b")),
    ("token_cost", (r"\btoken\s+costs?\b", r"\bcost\s+per\s+token\b", r"\btoken\s+spend\b")),
)


def detect_query_cost_constraints(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    if not text:
        return []

    rows: list[dict[str, Any]] = []
    for category, patterns in _CATEGORIES:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category})
    return sorted(rows, key=lambda row: row["category"])
