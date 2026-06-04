"""Detect source freshness intent in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CURRENT_RE = re.compile(r"\b(?:latest|current|today|now|recent|up[-\s]?to[-\s]?date|as\s+of\s+now|this\s+(?:week|month|year))\b", re.I)
_HISTORICAL_RE = re.compile(r"\b(?:historical|history|archived|in\s+(?:19|20)\d{2}|during\s+(?:19|20)\d{2})\b", re.I)
_POINT_RE = re.compile(r"\b(?:as\s+of|on)\s+((?:19|20)\d{2}-\d{2}-\d{2}|[A-Za-z]+\s+\d{1,2},?\s+(?:19|20)\d{2}|(?:19|20)\d{2})\b", re.I)
_EVERGREEN_RE = re.compile(r"\b(?:evergreen|timeless|conceptual|fundamentals|general\s+principles|how\s+does|what\s+is)\b", re.I)


def detect_query_source_freshness_intent(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    temporal_terms: list[str] = []
    point_match = _POINT_RE.search(normalized)
    for pattern in (_CURRENT_RE, _HISTORICAL_RE, _EVERGREEN_RE):
        temporal_terms.extend(match.group(0) for match in pattern.finditer(normalized))
    if point_match:
        temporal_terms.append(point_match.group(0))

    intent = _intent(normalized, point_match)
    requires_fresh = intent == "current"
    return {
        "intent": intent,
        "requires_fresh_sources": requires_fresh,
        "temporal_terms": sorted(set(temporal_terms), key=lambda value: value.casefold()),
        "suggested_source_date_filter": _date_filter(intent, point_match),
        "confidence": _confidence(intent),
        "normalized_query": normalized,
    }


def _intent(query: str, point_match: re.Match[str] | None) -> str:
    if point_match:
        return "point_in_time"
    if _CURRENT_RE.search(query):
        return "current"
    if _HISTORICAL_RE.search(query):
        return "historical"
    if _EVERGREEN_RE.search(query):
        return "evergreen"
    return "unspecified"


def _date_filter(intent: str, point_match: re.Match[str] | None) -> dict[str, str] | None:
    if intent == "current":
        return {"mode": "prefer_recent"}
    if intent == "historical":
        return {"mode": "allow_historical"}
    if intent == "point_in_time" and point_match:
        return {"mode": "as_of", "date": point_match.group(1)}
    if intent == "evergreen":
        return {"mode": "no_recency_boost"}
    return None


def _confidence(intent: str) -> float:
    return {"current": 0.86, "historical": 0.78, "point_in_time": 0.88, "evergreen": 0.7}.get(intent, 0.0)


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
