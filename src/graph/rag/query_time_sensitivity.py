"""Detect whether a query needs fresh context."""

from __future__ import annotations

import re
from typing import Any

_TERMS = ("latest", "current", "today", "recent", "forecast", "price", "schedule", "version", "changelog", "release")
_TERM_RE = {term: re.compile(r"\b" + re.escape(term) + r"s?\b", re.I) for term in _TERMS}


def detect_query_time_sensitivity(query: str) -> dict[str, Any]:
    text = str(query or "")
    matched = [term for term in _TERMS if _TERM_RE[term].search(text)]
    return {
        "requires_fresh_context": bool(matched),
        "matched_terms": matched,
        "suggested_recency_days": 30 if matched else None,
        "rationale": "Fresh context is needed for time-sensitive terms." if matched else "No freshness-sensitive terms detected.",
    }
