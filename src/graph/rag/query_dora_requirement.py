"""Detect DORA requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("dora", "high", (r"\bdigital\s+operational\s+resilience\s+act\b", r"\beu\s+dora\b", r"\bdora\s+(?:compliance|regulation|requirements?)\b")),
    ("ict_third_party_risk", "high", (r"\bict\s+third[- ]party\s+risk\b", r"\bthird[- ]party\s+ict\s+(?:providers?|risk)\b")),
    ("resilience_testing", "medium", (r"\boperational\s+resilience\s+testing\b", r"\bdigital\s+resilience\s+testing\b")),
    ("incident_reporting", "medium", (r"\bict\s+incident\s+reporting\b", r"\bmajor\s+ict\s+incidents?\b")),
    ("register_of_information", "medium", (r"\bregister\s+of\s+information\b",)),
    ("critical_provider", "medium", (r"\bcritical\s+ict\s+third[- ]party\s+providers?\b",)),
    ("financial_entity", "medium", (r"\bfinancial\s+entities?\b", r"\beu\s+financial\s+sector\b")),
)


def detect_query_dora_requirement(query: str) -> dict[str, Any]:
    matches = _matches(query)
    categories = sorted(dict.fromkeys(match["category"] for match in matches))
    return {"requires_dora": bool(matches), "categories": categories, "matches": matches}


def _matches(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _PATTERNS:
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                rows.append({"matched_text": match.group(0), "category": category, "severity": severity, "span": match.span()})
    return sorted(rows, key=lambda row: (row["span"][0], row["category"], row["matched_text"].casefold()))
