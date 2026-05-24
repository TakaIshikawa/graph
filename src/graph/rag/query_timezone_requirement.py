"""Detect timezone-aware interpretation needs in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_RELATIVE_TERMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("today", re.compile(r"\btoday\b", re.I)),
    ("yesterday", re.compile(r"\byesterday\b", re.I)),
    ("tomorrow", re.compile(r"\btomorrow\b", re.I)),
    ("market open", re.compile(r"\bmarket open\b", re.I)),
    ("market close", re.compile(r"\bmarket close\b", re.I)),
    ("close of business", re.compile(r"\bclose of business|cob\b", re.I)),
    ("local time", re.compile(r"\blocal time\b", re.I)),
)
_TZ_ABBREVIATION_RE = re.compile(r"\b(?:UTC|GMT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|CET|CEST|JST|AEST|AEDT)\b")
_UTC_OFFSET_RE = re.compile(r"\b(?:UTC|GMT)\s*[+-]\s*\d{1,2}(?::?\d{2})?\b", re.I)
_LOCATION_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("New York", re.compile(r"\bnew york\b|\bnyc\b", re.I)),
    ("London", re.compile(r"\blondon\b", re.I)),
    ("Tokyo", re.compile(r"\btokyo\b|\bjapan\b", re.I)),
    ("California", re.compile(r"\bcalifornia\b|\bsan francisco\b|\blos angeles\b", re.I)),
    ("United States", re.compile(r"\bunited states\b|\bu\.s\.\b", re.I)),
)


def detect_query_timezone_requirement(query: str) -> dict[str, Any]:
    """Return timezone cues, location hints, and normalization recommendations."""
    normalized = _normalize_query(query)
    original = " ".join(query.strip().split())
    relative_terms = [label for label, pattern in _RELATIVE_TERMS if pattern.search(normalized)]
    offsets = _normalize_offsets(original)
    abbreviation_text = original
    for offset in offsets:
        abbreviation_text = re.sub(re.escape(offset), " ", abbreviation_text, flags=re.I)
    explicit_timezones = sorted(set(_TZ_ABBREVIATION_RE.findall(abbreviation_text)).union(offsets))
    locations = [label for label, pattern in _LOCATION_HINTS if pattern.search(normalized)]
    requires = bool(relative_terms or explicit_timezones or locations)
    recommendations = []
    if relative_terms and not explicit_timezones:
        recommendations.append("resolve_relative_terms_against_user_or_query_timezone")
    if locations and not explicit_timezones:
        recommendations.append("map_location_hints_to_iana_timezones_before_date_math")
    if explicit_timezones:
        recommendations.append("normalize_times_to_utc_and_preserve_source_timezone")
    return {
        "requires_timezone_awareness": requires,
        "relative_time_terms": relative_terms,
        "explicit_timezones": explicit_timezones,
        "location_hints": locations,
        "normalization_recommendations": recommendations,
        "confidence": _confidence(relative_terms, explicit_timezones, locations),
        "normalized_query": normalized,
    }


def _normalize_offsets(text: str) -> set[str]:
    return {re.sub(r"\s+", "", match.group(0).upper()) for match in _UTC_OFFSET_RE.finditer(text)}


def _confidence(relative_terms: list[str], explicit_timezones: list[str], locations: list[str]) -> float:
    if explicit_timezones:
        return 0.9
    if relative_terms and locations:
        return 0.8
    if relative_terms:
        return 0.65
    if locations:
        return 0.45
    return 0.0


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.casefold().split())
