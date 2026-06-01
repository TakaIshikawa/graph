"""Detect accessibility accommodation requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("wcag", "high", re.compile(r"\b(?:wcag|web\s+content\s+accessibility\s+guidelines)\b", re.I)),
    ("screen_reader", "high", re.compile(r"\b(?:screen\s+reader|voiceover|nvda|jaws|assistive\s+technology)\b", re.I)),
    ("keyboard_only", "high", re.compile(r"\b(?:keyboard[-\s]only|keyboard\s+navigation|tab\s+order|focus\s+management)\b", re.I)),
    ("captions_transcripts", "medium", re.compile(r"\b(?:captions?|closed\s+captions?|subtitles?|transcripts?)\b", re.I)),
    ("alt_text", "medium", re.compile(r"\b(?:alt\s+text|alternative\s+text|image\s+descriptions?)\b", re.I)),
    ("contrast", "medium", re.compile(r"\b(?:color\s+contrast|colour\s+contrast|contrast\s+ratio|high\s+contrast)\b", re.I)),
    ("reduced_motion", "medium", re.compile(r"\b(?:reduced\s+motion|motion\s+sensitivity|disable\s+animations?|no\s+animations?)\b", re.I)),
)


def detect_query_accessibility_accommodation_requirements(query: str) -> list[dict[str, Any]]:
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
