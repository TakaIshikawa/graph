"""Detect subprocessor requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"\bsub[-\s]?processors?\b",
    r"\bthird[-\s]?party\s+processors?\b",
    r"\bdata\s+processing\s+(?:agreement|addendum|terms)\b",
    r"\bdpa\b",
    r"\bgdpr\b",
    r"\bpersonal\s+data\b",
    r"\bprivacy\b",
)

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "list_disclosure",
        "high",
        (
            r"\bsub[-\s]?processors?\s+(?:list|register|inventory)\b",
            r"\b(?:list|register|inventory)\s+of\s+sub[-\s]?processors?\b",
            r"\b(?:vendor|processor)\s+list\b",
            r"\b(?:disclose|disclosure\s+of)\s+(?:approved\s+)?sub[-\s]?processors?\b",
        ),
    ),
    (
        "notification",
        "high",
        (
            r"\bchange\s+notifications?\b",
            r"\bsub[-\s]?processor\s+notifications?\b",
            r"\b(?:advance|prior)\s+notice\s+(?:of|before)\s+(?:new\s+)?sub[-\s]?processors?\b",
            r"\bnotify\s+(?:us|customers?|controllers?)\s+(?:before|about|of)\s+(?:adding|changing|new)\s+sub[-\s]?processors?\b",
        ),
    ),
    (
        "objection_rights",
        "high",
        (
            r"\bobjection\s+rights?\b",
            r"\bright\s+to\s+object\b",
            r"\bobject\s+to\s+(?:new\s+)?sub[-\s]?processors?\b",
            r"\b\d+\s+days?\s+to\s+object\b",
            r"\bobjection\s+period\b",
        ),
    ),
    (
        "data_location",
        "medium",
        (
            r"\bsub[-\s]?processor\s+(?:locations?|countries|regions?)\b",
            r"\b(?:locations?|countries|regions?)\s+of\s+sub[-\s]?processors?\b",
            r"\bdata\s+(?:locations?|residency|processing\s+locations?)\b",
            r"\bwhere\s+sub[-\s]?processors?\s+(?:process|store|host)\b",
        ),
    ),
    (
        "onward_transfer",
        "high",
        (
            r"\bonward\s+(?:data\s+)?transfers?\b",
            r"\binternational\s+(?:data\s+)?transfers?\b",
            r"\bcross[-\s]?border\s+(?:data\s+)?transfers?\b",
            r"\bthird[-\s]?country\s+transfers?\b",
            r"\btransfers?\s+to\s+sub[-\s]?processors?\b",
        ),
    ),
    (
        "dpia_support",
        "medium",
        (
            r"\bdpia\s+(?:support|assistance|help)\b",
            r"\b(?:support|assist|assistance)\s+(?:with|for)\s+dpias?\b",
            r"\bdata\s+protection\s+impact\s+assessment\s+(?:support|assistance)\b",
            r"\b(?:support|assist|assistance)\s+(?:with|for)\s+data\s+protection\s+impact\s+assessments?\b",
        ),
    ),
)


def detect_query_subprocessor_requirements(query: str) -> list[dict[str, Any]]:
    """Return subprocessor requirement categories requested by a query."""

    text = str(query or "")
    if not _has_subprocessor_context(text):
        return []

    rows: list[dict[str, Any]] = []
    for category, severity, patterns in _CATEGORIES:
        match = _first_match(patterns, text)
        if match:
            rows.append(
                {
                    "matched_text": match.group(0),
                    "category": category,
                    "severity": severity,
                    "span": [match.start(), match.end()],
                }
            )
    return sorted(rows, key=lambda row: row["category"])


def _has_subprocessor_context(text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in _CONTEXT_PATTERNS)


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None
