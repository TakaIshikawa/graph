"""Detect platform-specific requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PLATFORM_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ios", re.compile(r"\b(?:ios|iphone|ipad)\b", re.I)),
    ("android", re.compile(r"\bandroid\b", re.I)),
    ("macos", re.compile(r"\b(?:macos|mac\s*os|os\s*x)\b", re.I)),
    ("windows", re.compile(r"\bwindows\b", re.I)),
    ("linux", re.compile(r"\blinux\b", re.I)),
    ("web", re.compile(r"\bweb(?:site|app)?\b", re.I)),
    ("browser", re.compile(r"\bbrowsers?\b", re.I)),
    ("chrome", re.compile(r"\bchrome\b", re.I)),
    ("safari", re.compile(r"\bsafari\b", re.I)),
    ("mobile", re.compile(r"\bmobile\b", re.I)),
    ("desktop", re.compile(r"\bdesktop\b", re.I)),
    ("api", re.compile(r"\bapi\b", re.I)),
    ("cli", re.compile(r"\b(?:cli|command[-\s]line|terminal)\b", re.I)),
    ("cloud", re.compile(r"\bcloud(?:[-\s]hosted)?\b", re.I)),
    ("self_hosted", re.compile(r"\bself[-\s]hosted\b", re.I)),
)

_AMBIGUOUS_PLATFORMS = {"mobile", "desktop", "browser", "web"}


def detect_query_platform_requirement(query: str) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    normalized = " ".join(query.casefold().split())
    platform_terms = _platform_terms(normalized)
    platforms = _platforms(platform_terms)
    ambiguity_flags = [f"broad_{platform}" for platform in platforms if platform in _AMBIGUOUS_PLATFORMS]
    return {
        "requires_platform_specificity": bool(platforms),
        "platforms": platforms,
        "platform_terms": platform_terms,
        "ambiguity_flags": ambiguity_flags,
        "confidence": _confidence(platforms, ambiguity_flags),
        "normalized_query": normalized,
    }


def _platform_terms(normalized_query: str) -> list[dict[str, Any]]:
    terms: list[dict[str, Any]] = []
    for platform, pattern in _PLATFORM_SPECS:
        for match in pattern.finditer(normalized_query):
            terms.append({"platform": platform, "term": match.group(0), "span": [match.start(), match.end()]})
    terms.sort(key=lambda row: (row["span"][0], row["span"][1], row["platform"]))
    return terms


def _platforms(platform_terms: list[dict[str, Any]]) -> list[str]:
    seen = {term["platform"] for term in platform_terms}
    return [platform for platform, _pattern in _PLATFORM_SPECS if platform in seen]


def _confidence(platforms: list[str], ambiguity_flags: list[str]) -> float:
    if not platforms:
        return 0.0
    if ambiguity_flags and len(ambiguity_flags) == len(platforms):
        return 0.55
    if ambiguity_flags:
        return 0.75
    return 0.9
