"""Detect update-cadence requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CADENCE: tuple[tuple[str, str | None, re.Pattern[str]], ...] = (
    ("daily", "P1D", re.compile(r"\bdaily\b|\bevery day\b", re.I)),
    ("weekly", "P7D", re.compile(r"\bweekly\b|\bevery week\b", re.I)),
    ("monthly", "P31D", re.compile(r"\bmonthly\b|\bevery month\b", re.I)),
    ("quarterly", "P92D", re.compile(r"\bquarterly\b|\bevery quarter\b", re.I)),
    ("annual", "P366D", re.compile(r"\bannual(?:ly)?\b|\byearly\b", re.I)),
    ("release_cadence", None, re.compile(r"\brelease cadence\b|\bupdate frequency\b", re.I)),
)
_REALTIME: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("real_time", re.compile(r"\breal[- ]time\b", re.I)),
    ("live", re.compile(r"\blive\b", re.I)),
    ("continuously_updated", re.compile(r"\bcontinuously updated\b|\bcontinuous updates\b", re.I)),
)
_GENERIC_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:today|yesterday|last year)\b", re.I)


def detect_query_update_frequency_requirement(query: str) -> dict[str, Any]:
    """Return update-cadence cues and conservative freshness hints."""
    normalized = _normalize_query(query)
    cadence_terms = [label for label, _, pattern in _CADENCE if pattern.search(normalized)]
    realtime = [label for label, pattern in _REALTIME if pattern.search(normalized)]
    requires = bool(cadence_terms or realtime)
    stale = _stale_hint(cadence_terms, realtime)
    recommendations = []
    if cadence_terms:
        recommendations.append("prefer_sources_with_declared_update_cadence")
    if realtime:
        recommendations.append("use_live_or_recently_refreshed_sources")
    return {
        "requires_cadence_awareness": requires,
        "cadence_terms": cadence_terms,
        "realtime_cues": realtime,
        "stale_if_older_than": stale,
        "recommendations": recommendations,
        "confidence": 0.85 if realtime else (0.7 if cadence_terms else 0.0),
        "normalized_query": normalized,
    }


def _stale_hint(cadence_terms: list[str], realtime: list[str]) -> str | None:
    if realtime:
        return "PT1H"
    for label, stale, _ in _CADENCE:
        if label in cadence_terms and stale is not None:
            return stale
    return None


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    normalized = " ".join(query.casefold().split())
    return normalized
