"""Detect retention policy requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_POLICY_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("data_retention", re.compile(r"\b(?:data\s+retention|retention\s+policy|retain(?:ed|ing)?|kept|keep\s+(?:data|records|files))\b", re.I)),
    ("deletion_window", re.compile(r"\b(?:delete|deletion|erase|erasure|purg(?:e|ed|ing)|right\s+to\s+(?:delete|erasure|be\s+forgotten))\b", re.I)),
    ("archival_period", re.compile(r"\b(?:archive|archived|archival|archiving|cold\s+storage)\b", re.I)),
    ("logs", re.compile(r"\b(?:log|logs|audit\s+trail|event\s+history)\b", re.I)),
    ("backups", re.compile(r"\b(?:backup|backups|snapshot|snapshots)\b", re.I)),
    ("expiry", re.compile(r"\b(?:expire|expires|expiry|expiration|ttl|time\s+to\s+live)\b", re.I)),
)
_WINDOW_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:hours?|hrs?|days?|weeks?|months?|years?|yrs?)\b",
    re.I,
)


def detect_query_retention_policy_requirement(query: str) -> dict[str, Any]:
    """Return retention policy terms, time windows, and matched phrases."""
    text = _normalize_query(query)
    matches = []
    for term, pattern in _POLICY_SPECS:
        for match in pattern.finditer(text):
            matches.append({"term": term, "phrase": match.group(0).strip(), "span": [match.start(), match.end()]})
    matches.sort(key=lambda row: (row["span"][0], row["span"][1], row["term"]))
    return {
        "requires_retention_policy": bool(matches),
        "retention_terms": sorted({row["term"] for row in matches}),
        "time_windows": _time_windows(text) if matches else [],
        "matched_phrases": _matched_phrases(matches),
    }


def _time_windows(text: str) -> list[str]:
    seen: set[str] = set()
    windows: list[str] = []
    for match in _WINDOW_RE.finditer(text):
        window = " ".join(match.group(0).split())
        key = window.casefold()
        if key not in seen:
            seen.add(key)
            windows.append(window)
    return windows


def _matched_phrases(matches: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    phrases: list[str] = []
    for match in matches:
        phrase = str(match["phrase"])
        key = phrase.casefold()
        if key not in seen:
            seen.add(key)
            phrases.append(phrase)
    return phrases


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.strip().split())
