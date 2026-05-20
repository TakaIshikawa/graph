"""Extract deterministic temporal intent signals from RAG queries."""

from __future__ import annotations

import re
from typing import Any

_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_YEAR_RE = re.compile(r"(?<![A-Za-z0-9])((?:18|19|20)\d{2})(?![A-Za-z0-9])")
_YEAR_RANGE_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:18|19|20)\d{2})\s*(?:-|to|through|until|and)\s*((?:18|19|20)\d{2})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_MONTH_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\b",
    re.IGNORECASE,
)

_MONTHS = {
    "jan": "january",
    "january": "january",
    "feb": "february",
    "february": "february",
    "mar": "march",
    "march": "march",
    "apr": "april",
    "april": "april",
    "may": "may",
    "jun": "june",
    "june": "june",
    "jul": "july",
    "july": "july",
    "aug": "august",
    "august": "august",
    "sep": "september",
    "sept": "september",
    "september": "september",
    "oct": "october",
    "october": "october",
    "nov": "november",
    "november": "november",
    "dec": "december",
    "december": "december",
}

_RECENCY_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("latest", re.compile(r"\blatest\b", re.IGNORECASE)),
    ("recent", re.compile(r"\brecent(?:ly)?\b", re.IGNORECASE)),
    ("newest", re.compile(r"\bnewest\b", re.IGNORECASE)),
    ("current", re.compile(r"\bcurrent(?:ly)?\b", re.IGNORECASE)),
    ("today", re.compile(r"\btoday\b", re.IGNORECASE)),
    ("this week", re.compile(r"\bthis\s+week\b", re.IGNORECASE)),
    ("last week", re.compile(r"\blast\s+week\b", re.IGNORECASE)),
    ("last month", re.compile(r"\blast\s+month\b", re.IGNORECASE)),
    ("last year", re.compile(r"\blast\s+year\b", re.IGNORECASE)),
)
_HISTORICAL_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("historical", re.compile(r"\bhistorical(?:ly)?\b", re.IGNORECASE)),
    ("history", re.compile(r"\bhistory\b", re.IGNORECASE)),
    ("archive", re.compile(r"\barchive(?:d|s)?\b", re.IGNORECASE)),
    ("past", re.compile(r"\bpast\b", re.IGNORECASE)),
    ("older", re.compile(r"\bolder\b", re.IGNORECASE)),
    ("before", re.compile(r"\bbefore\b", re.IGNORECASE)),
)


def extract_query_temporal_intent(query: str) -> dict[str, Any]:
    """Return normalized temporal constraints and cues for a query.

    The extractor is lexical and deterministic. It does not resolve relative
    dates against today's date; relative terms are classified as recency cues so
    callers can decide how to translate them for retrieval.
    """
    normalized_query = " ".join(str(query).split())
    if not normalized_query:
        return {
            "has_temporal_intent": False,
            "intent": "none",
            "confidence": 0.0,
            "years": [],
            "iso_dates": [],
            "months": [],
            "ranges": [],
            "recency_cues": [],
            "historical_cues": [],
            "reasons": [],
        }

    iso_dates = sorted({match.group(0) for match in _ISO_DATE_RE.finditer(normalized_query)})
    years = sorted({match.group(1) for match in _YEAR_RE.finditer(normalized_query)})
    months = sorted({_MONTHS[match.group(1).casefold()] for match in _MONTH_RE.finditer(normalized_query)})
    ranges = _ranges(normalized_query)
    recency_cues = _matched_cues(normalized_query, _RECENCY_CUES)
    historical_cues = _matched_cues(normalized_query, _HISTORICAL_CUES)

    reasons: list[str] = []
    if iso_dates:
        reasons.append("matched ISO date")
    if years:
        reasons.append("matched explicit year")
    if months:
        reasons.append("matched month name")
    if ranges:
        reasons.append("matched explicit range")
    if recency_cues:
        reasons.append("matched recency cue")
    if historical_cues:
        reasons.append("matched historical cue")

    intent = _intent(recency_cues, historical_cues, years, iso_dates, months, ranges)
    return {
        "has_temporal_intent": bool(reasons),
        "intent": intent,
        "confidence": _confidence(reasons, recency_cues, historical_cues, ranges),
        "years": years,
        "iso_dates": iso_dates,
        "months": months,
        "ranges": ranges,
        "recency_cues": recency_cues,
        "historical_cues": historical_cues,
        "reasons": reasons,
    }


def _matched_cues(query: str, cues: tuple[tuple[str, re.Pattern[str]], ...]) -> list[str]:
    return [label for label, pattern in cues if pattern.search(query)]


def _ranges(query: str) -> list[dict[str, str]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for match in _YEAR_RANGE_RE.finditer(query):
        start, end = match.group(1), match.group(2)
        if end < start:
            start, end = end, start
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"start": start, "end": end})
    return sorted(rows, key=lambda row: (row["start"], row["end"]))


def _intent(
    recency_cues: list[str],
    historical_cues: list[str],
    years: list[str],
    iso_dates: list[str],
    months: list[str],
    ranges: list[dict[str, str]],
) -> str:
    if recency_cues and not historical_cues:
        return "recency"
    if historical_cues and not recency_cues:
        return "historical"
    if ranges:
        return "range"
    if years or iso_dates or months:
        return "specific"
    if recency_cues and historical_cues:
        return "mixed"
    return "none"


def _confidence(
    reasons: list[str],
    recency_cues: list[str],
    historical_cues: list[str],
    ranges: list[dict[str, str]],
) -> float:
    if not reasons:
        return 0.0
    score = 0.35 + min(len(reasons), 4) * 0.12
    if ranges:
        score += 0.1
    if recency_cues or historical_cues:
        score += 0.08
    return round(min(score, 0.95), 2)
