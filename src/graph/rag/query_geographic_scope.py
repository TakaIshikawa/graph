"""Detect geographic scope constraints in user queries."""

from __future__ import annotations

import re
from typing import Any

_LOCATIONS = ("japan", "united states", "us", "usa", "europe", "eu", "uk", "canada", "australia", "china", "india")
_GLOBAL = re.compile(r"\b(worldwide|global(?:ly)?|international)\b", re.I)
_LOCAL = re.compile(r"\b(local|nearby|near me)\b", re.I)


def detect_query_geographic_scope(query: str) -> dict[str, Any]:
    text = str(query or "")
    matched_terms: list[str] = []
    locations: list[str] = []
    for location in _LOCATIONS:
        if re.search(rf"\b(?:in|for|within|across)?\s*{re.escape(location)}(?:[-\s]?only)?\b", text, re.I):
            locations.append(_normalize_location(location))
            matched_terms.append(location)
    global_scope = bool(_GLOBAL.search(text))
    local_scope = bool(_LOCAL.search(text))
    if global_scope:
        matched_terms.append(_GLOBAL.search(text).group(1).lower())  # type: ignore[union-attr]
    if local_scope:
        matched_terms.append(_LOCAL.search(text).group(1).lower())  # type: ignore[union-attr]
    scope_type = "none"
    if global_scope:
        scope_type = "global"
    elif local_scope:
        scope_type = "local"
    elif locations:
        scope_type = "location"
    return {
        "has_geographic_scope": bool(locations or global_scope or local_scope),
        "scope_type": scope_type,
        "locations": sorted(set(locations)),
        "global_scope": global_scope,
        "local_scope": local_scope,
        "matched_terms": matched_terms,
    }


def _normalize_location(location: str) -> str:
    aliases = {"us": "United States", "usa": "United States", "eu": "EU", "uk": "UK"}
    return aliases.get(location, location.title())
