"""Detect localization requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("language", "high", re.compile(r"\b(?:languages?|translate|translation|translated|multilingual|localized\s+copy|i18n)\b", re.I)),
    ("locale", "high", re.compile(r"\b(?:locale|locales|country[-\s]specific|region[-\s]specific|regional\s+settings)\b", re.I)),
    ("currency", "high", re.compile(r"\b(?:currenc(?:y|ies)|fx|exchange\s+rate|prices?\s+in\s+[a-z]{3})\b", re.I)),
    ("timezone", "high", re.compile(r"\b(?:time\s*zones?|timezone|local\s+time|utc|gmt)\b", re.I)),
    ("regional_format", "medium", re.compile(r"\b(?:date\s+format|number\s+format|regional\s+formatting|decimal\s+separator|address\s+format)\b", re.I)),
    ("units", "medium", re.compile(r"\b(?:measurement\s+units?|metric\s+units?|imperial\s+units?|celsius|fahrenheit|kilometers?|miles?)\b", re.I)),
)


def detect_query_localization_requirements(query: str) -> list[dict[str, Any]]:
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
