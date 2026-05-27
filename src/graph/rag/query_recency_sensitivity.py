"""Detect whether a query needs recent evidence."""

from __future__ import annotations

import re
from typing import Any

_RECENCY = {
    "latest": r"\blatest\b",
    "current": r"\bcurrent(?:ly)?\b",
    "today": r"\bas of today\b|\btoday\b",
    "now": r"\bnow\b",
    "recent_changes": r"\brecent changes?\b",
    "new": r"\bnew (?:rules?|polic(?:y|ies)|regulations?|changes?)\b",
}
_PAST_FIXED_RE = re.compile(r"\b(?:in|during|as of)\s+(?:19|20)\d{2}\b", re.I)


def detect_query_recency_sensitivity(query: str) -> dict[str, Any]:
    text = str(query or "")
    matches = [term for term, pattern in _RECENCY.items() if re.search(pattern, text, re.I)]
    if matches == ["current"] and _PAST_FIXED_RE.search(text):
        matches = []
    return {"requires_recent_evidence": bool(matches), "recency_terms": matches, "matched_phrases": matches.copy()}
