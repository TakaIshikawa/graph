"""Detect geographic requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_LOCAL_RE = re.compile(r"\b(?:near me|nearby|local(?:ly)?|in my area)\b", re.I)
_GLOBAL_RE = re.compile(r"\b(?:global|globally|worldwide|international)\b", re.I)
_REGION_RE = re.compile(r"\b(?:EU|European Union|APAC|EMEA|LATAM|North America|South America|Europe|Asia|Africa)\b", re.I)
_COUNTRY_RE = re.compile(r"\b(?:US|USA|U\.S\.|United States|UK|U\.K\.|United Kingdom|Canada|Japan|Germany|France|Australia|India|China)\b", re.I)
_CITY_PHRASE_RE = re.compile(r"\b(?:in|near|around|for|from)\s+([A-Z][A-Za-z]+(?:[\s-]+[A-Z][A-Za-z]+){0,3})\b")


def detect_query_geo_requirements(query: object) -> dict[str, Any]:
    normalized_query = " ".join(str(query or "").split())
    terms: list[str] = []
    scope_type = "none"

    local_terms = _matches(_LOCAL_RE, normalized_query)
    global_terms = _matches(_GLOBAL_RE, normalized_query)
    region_terms = [_normalize_geo(match.group(0)) for match in _REGION_RE.finditer(normalized_query)]
    country_terms = [_normalize_geo(match.group(0)) for match in _COUNTRY_RE.finditer(normalized_query)]
    city_terms = [_normalize_geo(match.group(1)) for match in _CITY_PHRASE_RE.finditer(normalized_query)]

    if local_terms:
        scope_type = "local"
    elif country_terms:
        scope_type = "country"
    elif region_terms:
        scope_type = "region"
    elif city_terms:
        scope_type = "city"
    elif global_terms:
        scope_type = "global"

    for term in local_terms + country_terms + region_terms + city_terms + global_terms:
        if term and term not in terms:
            terms.append(term)

    return {
        "has_geo_requirement": bool(terms),
        "geo_terms": terms,
        "scope_type": scope_type,
        "normalized_query": normalized_query,
    }


def _matches(pattern: re.Pattern[str], text: str) -> list[str]:
    return [_normalize_geo(match.group(0)) for match in pattern.finditer(text)]


def _normalize_geo(value: str) -> str:
    text = " ".join(value.replace(".", "").split())
    aliases = {
        "near me": "near me",
        "nearby": "nearby",
        "local": "local",
        "locally": "local",
        "in my area": "in my area",
        "us": "US",
        "usa": "US",
        "united states": "United States",
        "uk": "UK",
        "united kingdom": "United Kingdom",
        "eu": "EU",
        "european union": "European Union",
    }
    return aliases.get(text.casefold(), text)
