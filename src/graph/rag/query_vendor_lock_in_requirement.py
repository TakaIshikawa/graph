"""Detect vendor lock-in requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("lock_in", "high", re.compile(r"\b(?:vendor\s+lock[-\s]?in|lock[-\s]?in|locked\s+in|avoid\s+lock[-\s]?in)\b", re.I)),
    ("proprietary_dependency", "high", re.compile(r"\b(?:proprietary\s+(?:dependency|dependencies|apis?|formats?|features?|services?)|depends?\s+on\s+proprietary|closed\s+source\s+dependency)\b", re.I)),
    ("exit_strategy", "high", re.compile(r"\b(?:exit\s+strateg(?:y|ies)|vendor\s+exit|offboarding\s+plan|escape\s+hatch)\b", re.I)),
    ("migration_path", "medium", re.compile(r"\b(?:migration\s+path|migration\s+plan|path\s+to\s+migrate|migrate\s+away|move\s+off\s+(?:the\s+)?vendor)\b", re.I)),
    ("switching_cost", "medium", re.compile(r"\b(?:switching\s+costs?|cost\s+to\s+switch|switch\s+vendors?|vendor\s+switching|replacement\s+costs?)\b", re.I)),
)


def detect_query_vendor_lock_in_requirements(query: str) -> list[dict[str, Any]]:
    normalized = _normalize_query(query)
    rows = []
    for category, severity, pattern in _CATEGORY_SPECS:
        match = pattern.search(normalized)
        if match:
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity, "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: (row["span"][0], row["span"][1], row["category"]))
    return rows


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").casefold().split())
