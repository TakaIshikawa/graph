"""Detect dataset-style evidence requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENT_FLAGS = (
    "requires_raw_data",
    "requires_tabular_data",
    "requires_benchmark_data",
    "requires_methodology_data",
    "requires_downloadable_source",
)

_CUES: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "requires_raw_data": (
        ("raw data", re.compile(r"\braw\s+data\b", re.IGNORECASE)),
        ("dataset", re.compile(r"\bdatasets?\b", re.IGNORECASE)),
        ("source data", re.compile(r"\bsource\s+data\b", re.IGNORECASE)),
        ("data file", re.compile(r"\bdata\s+files?\b", re.IGNORECASE)),
        ("microdata", re.compile(r"\bmicrodata\b", re.IGNORECASE)),
    ),
    "requires_tabular_data": (
        ("table", re.compile(r"\btables?\b", re.IGNORECASE)),
        ("csv", re.compile(r"\bcsvs?\b", re.IGNORECASE)),
        ("spreadsheet", re.compile(r"\bspreadsheet\b", re.IGNORECASE)),
        ("rows", re.compile(r"\brows?\b", re.IGNORECASE)),
        ("columns", re.compile(r"\bcolumns?\b", re.IGNORECASE)),
    ),
    "requires_benchmark_data": (
        ("benchmark", re.compile(r"\bbenchmarks?\b", re.IGNORECASE)),
        ("leaderboard", re.compile(r"\bleaderboards?\b", re.IGNORECASE)),
        ("baseline", re.compile(r"\bbaselines?\b", re.IGNORECASE)),
        ("test set", re.compile(r"\btest\s+sets?\b", re.IGNORECASE)),
        ("evaluation dataset", re.compile(r"\bevaluation\s+datasets?\b", re.IGNORECASE)),
    ),
    "requires_methodology_data": (
        ("sample size", re.compile(r"\bsample\s+sizes?\b|\bn\s*=\s*\d+\b", re.IGNORECASE)),
        ("participants", re.compile(r"\bparticipants?\b", re.IGNORECASE)),
        ("respondents", re.compile(r"\brespondents?\b", re.IGNORECASE)),
        ("methodology", re.compile(r"\bmethodolog(?:y|ies|ical)\b", re.IGNORECASE)),
        ("methods appendix", re.compile(r"\bmethods?\s+appendix\b", re.IGNORECASE)),
        ("appendix", re.compile(r"\bappendix\b", re.IGNORECASE)),
        ("survey instrument", re.compile(r"\bsurvey\s+instrument\b", re.IGNORECASE)),
        ("codebook", re.compile(r"\bcodebooks?\b", re.IGNORECASE)),
    ),
    "requires_downloadable_source": (
        ("download", re.compile(r"\bdownload(?:able|s|ed|ing)?\b", re.IGNORECASE)),
        ("link to file", re.compile(r"\blink\s+to\s+(?:the\s+)?files?\b", re.IGNORECASE)),
        ("repository", re.compile(r"\brepositor(?:y|ies)\b", re.IGNORECASE)),
        ("github", re.compile(r"\bgithub\b", re.IGNORECASE)),
        ("data portal", re.compile(r"\bdata\s+portal\b", re.IGNORECASE)),
    ),
}

_DIRECT_DATASET_FLAGS = {
    "requires_raw_data",
    "requires_tabular_data",
    "requires_benchmark_data",
    "requires_downloadable_source",
}


def detect_query_dataset_requirements(query: str) -> dict[str, Any]:
    """Return dataset-style evidence requirement flags for a RAG query."""
    normalized_query = _normalize_query(query)
    matches = {
        flag: [label for label, pattern in cues if pattern.search(normalized_query)]
        for flag, cues in _CUES.items()
    }
    flags = {flag: bool(matches[flag]) for flag in _REQUIREMENT_FLAGS}

    return {
        "normalized_query": normalized_query,
        **flags,
        "matched_terms": _flatten_matches(matches),
        "requirement_matches": matches,
        "confidence": _confidence(flags, matches),
    }


def _confidence(flags: dict[str, bool], matches: dict[str, list[str]]) -> float:
    category_count = sum(1 for value in flags.values() if value)
    if category_count == 0:
        return 0.0

    term_count = sum(len(terms) for terms in matches.values())
    direct_category_count = sum(1 for flag in _DIRECT_DATASET_FLAGS if flags[flag])
    score = 0.32 + (category_count * 0.13) + (term_count * 0.04) + (direct_category_count * 0.08)
    if flags["requires_methodology_data"] and not direct_category_count:
        score -= 0.04
    return round(min(score, 0.99), 3)


def _flatten_matches(matches: dict[str, list[str]]) -> list[str]:
    seen: set[str] = set()
    flattened: list[str] = []
    for flag in _REQUIREMENT_FLAGS:
        for term in matches[flag]:
            if term not in seen:
                seen.add(term)
                flattened.append(term)
    return flattened


def _normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a non-empty string")
    normalized = " ".join(query.strip().split())
    if not normalized:
        raise ValueError("query must be a non-empty string")
    return normalized
