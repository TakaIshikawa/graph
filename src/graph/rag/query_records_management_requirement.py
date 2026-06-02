"""Detect records-management requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("records_management", (r"\brecords\s+management\b",)),
    ("recordkeeping", (r"\brecordkeeping\b", r"\brecord\s+keeping\b")),
    ("records_schedule", (r"\brecords?\s+schedule\b", r"\bretention\s+schedule\b")),
    ("disposition", (r"\brecords?\s+disposition\b", r"\bdisposition\s+schedule\b")),
    ("archive_policy", (r"\barchive\s+policy\b", r"\barchiving\s+policy\b")),
    ("official_record", (r"\bofficial\s+records?\b", r"\brecord\s+designation\b")),
)


def detect_query_records_management_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    categories = [category for category, patterns in _CATEGORIES if _first_match(patterns, text)]
    return {"requires_records_management": bool(categories), "cue_categories": categories}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    text = " ".join(str(query or "").split())
    if not text:
        raise ValueError("query must not be empty")
    return text
