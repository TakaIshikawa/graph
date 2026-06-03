"""Detect Sarbanes-Oxley requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("sox", "high", (r"\bsox\b", r"\bsarbanes[- ]oxley\b")),
    ("section_404", "high", (r"\bsection\s+404\b", r"\bsox\s*404\b")),
    ("icfr", "high", (r"\bicfr\b", r"\binternal\s+controls?\s+over\s+financial\s+reporting\b")),
    ("segregation_of_duties", "medium", (r"\bsegregation\s+of\s+duties\b", r"\bseparation\s+of\s+duties\b")),
    ("audit_evidence", "medium", (r"\baudit\s+evidence\b", r"\bcontrol\s+evidence\b")),
    ("financial_change_control", "medium", (r"\bfinancial\s+systems?\s+change\s+control\b", r"\bchange\s+control\s+for\s+financial\s+systems?\b")),
)


def detect_query_sox_requirement(query: str) -> dict[str, Any]:
    matches = _matches(query)
    categories = sorted(dict.fromkeys(match["category"] for match in matches))
    return {"requires_sox": bool(matches), "categories": categories, "matches": matches}


def _matches(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _PATTERNS:
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                rows.append({"matched_text": match.group(0), "category": category, "severity": severity, "span": match.span()})
    return sorted(rows, key=lambda row: (row["span"][0], row["category"], row["matched_text"].casefold()))
