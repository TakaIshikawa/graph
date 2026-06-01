"""Detect backward compatibility requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("backward_compatible", "medium", re.compile(r"\b(?:backward[-\s]compatible|backwards[-\s]compatible|backward\s+compatibility|backwards\s+compatibility)\b", re.I)),
    ("legacy_support", "medium", re.compile(r"\b(?:legacy\s+support|support\s+legacy|legacy\s+(?:clients?|systems?|versions?|apis?)|legacy-compatible)\b", re.I)),
    ("breaking_change", "high", re.compile(r"\b(?:breaking\s+changes?|non[-\s]breaking|avoid\s+breaking|break\s+(?:existing|current)\s+(?:clients?|apis?|integrations?))\b", re.I)),
    ("compatibility_matrix", "medium", re.compile(r"\b(?:compatibility\s+matrix|support\s+matrix|version\s+matrix|client\s+compatibility)\b", re.I)),
    ("older_clients", "medium", re.compile(r"\b(?:older\s+clients?|old\s+clients?|previous\s+clients?|existing\s+clients?|older\s+versions?)\b", re.I)),
)


def detect_query_backward_compatibility_requirements(query: str) -> list[dict[str, Any]]:
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
