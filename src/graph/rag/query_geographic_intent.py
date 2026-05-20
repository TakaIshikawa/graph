"""Detect geographic constraints in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_EXPLICIT_LOCATION_RE = re.compile(
    r"\b(?:in|near|around|within|from|for)\s+([A-Z][A-Za-z]*(?:[\s-]+[A-Z][A-Za-z]*){0,3}|US|USA|U\.S\.|UK|U\.K\.|EU)\b"
)
_KNOWN_REGION_RE = re.compile(
    r"\b(US|USA|U\.S\.|United\s+States|UK|U\.K\.|United\s+Kingdom|EU|European\s+Union|APAC|EMEA|LATAM)\b",
    re.IGNORECASE,
)
_LOCAL_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("near", re.compile(r"\bnear(?:by)?\b", re.IGNORECASE)),
    ("local", re.compile(r"\blocal(?:ly)?\b", re.IGNORECASE)),
    ("in my area", re.compile(r"\bin\s+my\s+area\b", re.IGNORECASE)),
    ("within", re.compile(r"\bwithin\s+\d+\s*(?:km|mi|miles|kilometers)\b", re.IGNORECASE)),
)
_GLOBAL_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("global", re.compile(r"\bglobal(?:ly)?|worldwide|international\b", re.IGNORECASE)),
    ("remote", re.compile(r"\bremote\b", re.IGNORECASE)),
    ("anywhere", re.compile(r"\banywhere\b", re.IGNORECASE)),
)
_COMPARISON_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("between locations", re.compile(r"\bbetween\s+.+?\s+and\s+.+", re.IGNORECASE)),
    ("versus", re.compile(r"\b(?:vs\.?|versus)\b", re.IGNORECASE)),
    ("compare", re.compile(r"\bcompar(?:e|ing|ison)\b", re.IGNORECASE)),
    ("by region", re.compile(r"\bby\s+(?:country|region|city|state|market)\b", re.IGNORECASE)),
)


def detect_query_geographic_intent(query: str) -> dict[str, Any]:
    """Return stable geographic-intent flags and extracted location-like terms."""
    normalized_query = " ".join(str(query).split())
    reasons = {
        "local": _matched_cues(normalized_query, _LOCAL_CUES),
        "global_remote": _matched_cues(normalized_query, _GLOBAL_CUES),
        "explicit_location": [],
        "location_comparison": _matched_cues(normalized_query, _COMPARISON_CUES),
    }
    terms = _location_terms(normalized_query)
    if terms:
        reasons["explicit_location"] = ["location phrase"]

    return {
        "normalized_query": normalized_query,
        "local": bool(reasons["local"]),
        "global_remote": bool(reasons["global_remote"]),
        "explicit_location": bool(terms),
        "location_comparison": bool(reasons["location_comparison"]),
        "location_like_terms": terms,
        "reasons": reasons,
    }


def _matched_cues(query: str, cues: tuple[tuple[str, re.Pattern[str]], ...]) -> list[str]:
    return [label for label, pattern in cues if pattern.search(query)]


def _location_terms(query: str) -> list[str]:
    terms: set[str] = set()
    for match in _EXPLICIT_LOCATION_RE.finditer(query):
        terms.add(_normalize_location(match.group(1)))
    for match in _KNOWN_REGION_RE.finditer(query):
        terms.add(_normalize_location(match.group(1)))
    return sorted(term for term in terms if term)


def _normalize_location(value: str) -> str:
    text = " ".join(value.replace(".", "").split())
    aliases = {
        "u s": "US",
        "us": "US",
        "usa": "US",
        "united states": "United States",
        "u k": "UK",
        "uk": "UK",
        "united kingdom": "United Kingdom",
        "eu": "EU",
        "european union": "European Union",
        "apac": "APAC",
        "emea": "EMEA",
        "latam": "LATAM",
    }
    return aliases.get(text.casefold(), text)
