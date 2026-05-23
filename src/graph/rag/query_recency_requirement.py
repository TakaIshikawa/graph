"""Detect freshness requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_EXPLICIT_WINDOWS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("last 7 days", "P7D", re.compile(r"\blast\s+7\s+days?\b", re.IGNORECASE)),
    ("last week", "P7D", re.compile(r"\blast\s+week\b", re.IGNORECASE)),
    ("this week", "P7D", re.compile(r"\bthis\s+week\b", re.IGNORECASE)),
    ("this month", "P1M", re.compile(r"\bthis\s+month\b", re.IGNORECASE)),
    ("last month", "P1M", re.compile(r"\blast\s+month\b", re.IGNORECASE)),
    ("this year", "P1Y", re.compile(r"\bthis\s+year\b", re.IGNORECASE)),
)
_VAGUE_TERMS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("current", "current", re.compile(r"\bcurrent(?:ly)?\b", re.IGNORECASE)),
    ("latest", "latest", re.compile(r"\blatest\b", re.IGNORECASE)),
    ("newest", "latest", re.compile(r"\bnewest\b", re.IGNORECASE)),
    ("recently", "recent", re.compile(r"\brecent(?:ly)?\b", re.IGNORECASE)),
    ("today", "P1D", re.compile(r"\btoday\b", re.IGNORECASE)),
)


def detect_query_recency_requirement(query: str) -> dict[str, Any]:
    """Return deterministic recency signals for a retrieval query."""
    normalized = _inline_text(query)
    matched_terms: list[str] = []
    window = "none"

    for term, candidate_window, pattern in _EXPLICIT_WINDOWS:
        if pattern.search(normalized):
            matched_terms.append(term)
            if window == "none":
                window = candidate_window

    for term, candidate_window, pattern in _VAGUE_TERMS:
        if pattern.search(normalized):
            matched_terms.append(term)
            if window == "none":
                window = candidate_window

    requires = bool(matched_terms)
    retrieval_hint = (
        f"prefer sources matching recency window {window}"
        if requires and window not in {"current", "latest", "recent"}
        else "prefer the most recently updated authoritative sources"
        if requires
        else "no recency preference detected"
    )
    return {
        "query": normalized,
        "requires_recency": requires,
        "recency_window": window,
        "matched_terms": matched_terms,
        "retrieval_hint": retrieval_hint,
    }


def _inline_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())
