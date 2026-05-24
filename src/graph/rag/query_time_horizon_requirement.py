"""Detect time horizon requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
_YEAR_RANGE_RE = re.compile(r"\b((?:18|19|20)\d{2})\s*(?:-|to|through|until)\s*((?:18|19|20)\d{2})\b", re.I)
_DATE_RANGE_RE = re.compile(
    r"\b(?:from|between)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|(?:18|19|20)\d{2})"
    r"\s+(?:to|and|through|until)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|(?:18|19|20)\d{2})\b",
    re.I,
)

_FORECAST_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("forecast", re.compile(r"\bforecast(?:s|ed|ing)?\b", re.I)),
    ("predict", re.compile(r"\bpredict(?:ion|ions|ed|ing)?\b", re.I)),
    ("projection", re.compile(r"\bproject(?:ion|ions|ed|ing)?\b", re.I)),
    ("next", re.compile(r"\bnext\s+(?:week|month|quarter|year|decade|\d+\s+(?:days|weeks|months|years))\b", re.I)),
    ("future", re.compile(r"\bfuture\b", re.I)),
)
_HISTORICAL_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("historical", re.compile(r"\bhistorical(?:ly)?\b", re.I)),
    ("past", re.compile(r"\bpast\s+(?:week|month|quarter|year|decade|\d+\s+(?:days|weeks|months|years))\b", re.I)),
    ("history", re.compile(r"\bhistory\b", re.I)),
    ("since", re.compile(r"\bsince\s+(?:\d{4}|last\s+\w+)\b", re.I)),
    ("over time", re.compile(r"\bover\s+time\b", re.I)),
)
_CURRENT_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("current", re.compile(r"\bcurrent(?:ly)?\b", re.I)),
    ("now", re.compile(r"\bnow\b", re.I)),
    ("today", re.compile(r"\btoday\b", re.I)),
    ("as of", re.compile(r"\bas\s+of\b", re.I)),
    ("latest", re.compile(r"\blatest\b", re.I)),
)


def detect_query_time_horizon_requirement(query: str) -> dict[str, Any]:
    """Return deterministic time horizon requirements for a query."""
    normalized = " ".join(str(query or "").split())
    ranges = _date_ranges(normalized)
    matched = {
        "forecast": _matched_terms(normalized, _FORECAST_CUES),
        "historical": _matched_terms(normalized, _HISTORICAL_CUES),
        "current": _matched_terms(normalized, _CURRENT_CUES),
        "bounded_range": [row["text"] for row in ranges],
    }
    horizon_types = [name for name, terms in matched.items() if terms]
    confidence = 0.0 if not horizon_types else round(min(0.55 + 0.12 * len(horizon_types) + 0.08 * len(ranges), 0.95), 2)
    return {
        "query": normalized,
        "requires_time_horizon": bool(horizon_types),
        "horizon_types": horizon_types,
        "matched_terms": matched,
        "date_ranges": ranges,
        "confidence": confidence,
    }


def _matched_terms(query: str, cues: tuple[tuple[str, re.Pattern[str]], ...]) -> list[str]:
    return [label for label, pattern in cues if pattern.search(query)]


def _date_ranges(query: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in _DATE_RANGE_RE.finditer(query):
        row = {"start": match.group(1).strip(), "end": match.group(2).strip(), "text": match.group(0).strip()}
        key = (row["start"].casefold(), row["end"].casefold(), row["text"].casefold())
        if key not in seen:
            seen.add(key)
            rows.append(row)
    for match in _YEAR_RANGE_RE.finditer(query):
        start, end = match.group(1), match.group(2)
        row = {"start": min(start, end), "end": max(start, end), "text": match.group(0).strip()}
        key = (row["start"], row["end"], row["text"].casefold())
        if key not in seen and not any(existing["start"] == row["start"] and existing["end"] == row["end"] for existing in rows):
            seen.add(key)
            rows.append(row)
    if len(_ISO_DATE_RE.findall(query)) >= 2 and not rows:
        dates = _ISO_DATE_RE.findall(query)
        rows.append({"start": dates[0], "end": dates[1], "text": f"{dates[0]} to {dates[1]}"})
    years = _YEAR_RE.findall(query)
    if len(years) >= 2 and not rows:
        rows.append({"start": min(years[0], years[1]), "end": max(years[0], years[1]), "text": f"{years[0]} to {years[1]}"})
    return rows
