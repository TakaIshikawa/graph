"""Detect rate-limit backoff requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_RATE_LIMIT_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"\bapi\b",
    r"\bhttp\b",
    r"\bendpoints?\b",
    r"\brequests?\b",
    r"\bsdk\b",
    r"\bintegration\b",
    r"\brate[-\s]?limits?\b",
    r"\bthrottl(?:e|ed|es|ing)\b",
    r"\bquota\b",
)
_BACKOFF_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("exponential_backoff", (r"\bexponential\s+backoff\b", r"\bback\s+off\s+exponentially\b")),
    ("backoff_strategy", (r"\bbackoff\s+(?:strategy|policy|schedule|delay|delays)\b", r"\bretry\s+backoff\b")),
    ("jitter", (r"\bjitter\b", r"\brandomi[sz]ed\s+(?:delay|backoff)\b", r"\brandom\s+(?:delay|backoff)\b")),
    ("throttling_recovery", (r"\bthrottling\s+recovery\b", r"\brecover(?:y|ing)?\s+from\s+throttl(?:ing|es?)\b")),
)
_RETRY_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("http_429", (r"\bhttp\s+429\b", r"\b429\b", r"\btoo\s+many\s+requests\b")),
    ("retry_after", (r"\bretry[-\s]?after\b", r"\bretry\s+after\s+headers?\b")),
    ("idempotent_retries", (r"\bidempotent\s+retr(?:y|ies)\b", r"\bretr(?:y|ies)\s+idempotent\s+requests?\b")),
    ("retry_policy", (r"\bretry\s+(?:policy|logic|behavior|handling|strategy)\b", r"\bhandling\s+retries\b")),
)


def detect_query_rate_limit_backoff_requirement(query: str) -> dict[str, Any]:
    """Return rate-limit retry and backoff requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    backoff_matches = _matches(_BACKOFF_SPECS, text)
    retry_matches = _matches(_RETRY_SPECS, text)
    backoff_terms = [match["term"] for match in backoff_matches]
    retry_terms = [match["term"] for match in retry_matches]
    requires_backoff = _requires_backoff(text, backoff_terms, retry_terms)

    return {
        "requires_rate_limit_backoff": requires_backoff,
        "backoff_terms": backoff_terms,
        "retry_terms": retry_terms,
        "matched_phrases": _matched_phrases(backoff_matches + retry_matches),
        "recommendations": _recommendations(requires_backoff),
        "confidence": _confidence(requires_backoff, backoff_terms, retry_terms, text),
    }


def _matches(specs: tuple[tuple[str, tuple[str, ...]], ...], text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for term, patterns in specs:
        match = _first_match(patterns, text)
        if match:
            rows.append({"term": term, "matched_text": match.group(0), "span": (match.start(), match.end())})
    return sorted(rows, key=lambda row: (row["span"][0], row["term"]))


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _requires_backoff(text: str, backoff_terms: list[str], retry_terms: list[str]) -> bool:
    if not text:
        return False
    strong_retry = bool({"http_429", "retry_after", "idempotent_retries"} & set(retry_terms))
    return bool(backoff_terms and (retry_terms or _has_rate_limit_context(text))) or strong_retry


def _has_rate_limit_context(text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in _RATE_LIMIT_CONTEXT_PATTERNS)


def _matched_phrases(matches: list[dict[str, Any]]) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for match in sorted(matches, key=lambda row: (row["span"][0], row["term"])):
        phrase = str(match["matched_text"])
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            phrases.append(phrase)
    return phrases


def _recommendations(requires_backoff: bool) -> list[str]:
    if not requires_backoff:
        return []
    return [
        "Honor Retry-After headers before retrying rate-limited requests.",
        "Use bounded exponential backoff with jitter to avoid synchronized retry spikes.",
        "Limit automatic retries to idempotent operations or requests with idempotency keys.",
    ]


def _confidence(requires_backoff: bool, backoff_terms: list[str], retry_terms: list[str], text: str) -> str:
    if not requires_backoff:
        return "low" if backoff_terms or retry_terms else "none"
    if backoff_terms and retry_terms:
        return "high"
    if {"http_429", "retry_after"} & set(retry_terms) or _has_rate_limit_context(text):
        return "medium"
    return "low"
