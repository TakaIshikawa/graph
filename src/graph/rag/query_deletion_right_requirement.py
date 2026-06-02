"""Detect deletion-right and erasure workflow requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("right_to_deletion", (r"\bright\s+to\s+deletion\b", r"\bright\s+to\s+delete\b")),
    ("right_to_erasure", (r"\bright\s+to\s+erasure\b", r"\berasure\s+right\b")),
    ("account_deletion", (r"\baccount\s+deletion\b", r"\bdelete\s+(?:a\s+)?(?:user\s+)?account\b")),
    ("data_deletion_request", (r"\bdata\s+deletion\s+request\b", r"\bdeletion\s+request\b", r"\berasure\s+request\b")),
    (
        "purge_timeline",
        (r"\bpurge\s+timelines?\b", r"\bpurg(?:e|ed|ing)\s+(?:within|after)\b", r"\bdelete\s+\w+\s+within\b"),
    ),
    ("hard_delete", (r"\bhard\s+delete\b", r"\bpermanent(?:ly)?\s+delete\b")),
    ("soft_delete", (r"\bsoft\s+delete\b", r"\bsoft-deleted\b")),
    ("retention_after_deletion", (r"\bretention\s+after\s+deletion\b", r"\bretain(?:ed)?\s+after\s+(?:account\s+)?deletion\b")),
)
_DURATION_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:hours?|hrs?|days?|weeks?|months?|years?|yrs?)\b", re.I)


def detect_query_deletion_right_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    categories = [category for category, patterns in _CATEGORIES if _first_match(patterns, text)]
    return {
        "requires_deletion_right": bool(categories),
        "cue_categories": categories,
        "timing_values": _duration_mentions(text) if categories else [],
    }


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _duration_mentions(text: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for match in _DURATION_RE.finditer(text):
        value = " ".join(match.group(0).split())
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            values.append(value)
    return values


def _normalize_query(query: str) -> str:
    text = " ".join(str(query or "").split())
    if not text:
        raise ValueError("query must not be empty")
    return text
