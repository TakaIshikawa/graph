"""Detect mobile platform support requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CUE_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ios", re.compile(r"\b(?:ios|iphone)\b", re.I)),
    ("android", re.compile(r"\bandroid\b", re.I)),
    ("mobile_app", re.compile(r"\bmobile\s+app(?:lication)?\b", re.I)),
    ("responsive_mobile_ui", re.compile(r"\b(?:responsive\s+mobile\s+ui|mobile\s+responsive|responsive\s+on\s+mobile)\b", re.I)),
    ("tablet_support", re.compile(r"\b(?:tablet\s+support|ipad|ipados|android\s+tablet)\b", re.I)),
    ("app_store_availability", re.compile(r"\b(?:app\s+store|play\s+store|google\s+play)\b", re.I)),
    ("mobile_browser", re.compile(r"\bmobile\s+browser\b", re.I)),
    ("minimum_os_version", re.compile(r"\b(?:minimum\s+(?:os|ios|android)\s+version|min(?:imum)?\s+supported\s+os)\b", re.I)),
)

_VALUE_PATTERN = re.compile(r"\b(?:ios\s*\d+(?:\.\d+)?|android\s*\d+(?:\.\d+)?|ipados(?:\s*\d+(?:\.\d+)?)?|play\s+store|google\s+play|app\s+store)\b", re.I)


def detect_query_mobile_support_requirement(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    cue_matches = _cue_matches(normalized)
    return {
        "requires_mobile_support": bool(cue_matches),
        "cue_categories": [match["category"] for match in cue_matches],
        "matched_cues": cue_matches,
        "platform_values": _platform_values(normalized),
        "normalized_query": normalized,
    }


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.split())


def _cue_matches(normalized_query: str) -> list[dict[str, Any]]:
    rows = []
    for category, pattern in _CUE_SPECS:
        match = pattern.search(normalized_query)
        if match:
            rows.append({"category": category, "matched_text": match.group(0), "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: (row["span"][0], row["span"][1], row["category"]))
    return rows


def _platform_values(normalized_query: str) -> list[str]:
    seen: set[str] = set()
    values = []
    for match in _VALUE_PATTERN.finditer(normalized_query):
        value = match.group(0)
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            values.append(value)
    return values
