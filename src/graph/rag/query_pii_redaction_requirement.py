"""Detect PII redaction requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RADIUS = 36

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "redaction",
        (
            r"\bredact(?:ion|ed|ing)?\s+(?:of\s+)?(?:pii|personal\s+data|personal\s+information|personally\s+identifiable\s+information)\b",
            r"\b(?:pii|personal\s+data|personal\s+information|personally\s+identifiable\s+information)\s+redact(?:ion|ed|ing)?\b",
        ),
    ),
    (
        "masking",
        (
            r"\bmask(?:ing|ed)?\s+(?:pii|personal\s+data|personal\s+information|sensitive\s+fields?)\b",
            r"\b(?:pii|personal\s+data|personal\s+information|sensitive\s+fields?)\s+mask(?:ing|ed)?\b",
        ),
    ),
    (
        "anonymization",
        (
            r"\banonymi[sz](?:e|ation|ed|ing)\b",
            r"\bde[-\s]?identif(?:y|ication|ied|ying)\b",
        ),
    ),
    (
        "pseudonymization",
        (
            r"\bpseudonymi[sz](?:e|ation|ed|ing)\b",
            r"\bpseudonymous\s+(?:identifiers?|data|users?)\b",
        ),
    ),
    (
        "tokenization",
        (
            r"\btokeni[sz]ation\b",
            r"\btokeni[sz](?:e|ation|ed|ing)\s+(?:pii|personal\s+data|personal\s+information|sensitive\s+fields?|identifiers?|emails?)\b",
            r"\b(?:pii|personal\s+data|personal\s+information|sensitive\s+fields?|identifiers?|emails?)\s+tokeni[sz](?:e|ation|ed|ing)\b",
        ),
    ),
    (
        "sensitive_fields",
        (
            r"\bsensitive\s+fields?\b",
            r"\bsensitive\s+(?:columns?|attributes?|properties?)\b",
        ),
    ),
    (
        "log_redaction",
        (
            r"\bredact(?:ion|ed|ing)?\s+(?:pii|personal\s+data|personal\s+information|sensitive\s+fields?)\s+(?:from|in)\s+logs?\b",
            r"\blogs?\s+(?:pii|personal\s+data|personal\s+information|sensitive\s+fields?)\s+redact(?:ion|ed|ing)?\b",
            r"\blogs?\s+redaction\b",
            r"\bredacted\s+logs?\b",
        ),
    ),
    (
        "data_minimization",
        (
            r"\bdata\s+minimi[sz]ation\b",
            r"\bminimi[sz]e\s+(?:pii|personal\s+data|personal\s+information)\b",
        ),
    ),
)


def detect_pii_redaction_requirements(query: str) -> list[dict[str, Any]]:
    """Return PII redaction, masking, and minimization cues mentioned by a query."""
    text = _normalize_query(query)
    if not text:
        return []

    rows: list[dict[str, Any]] = []
    for category, patterns in _CATEGORIES:
        match = _first_match(patterns, text)
        if match:
            rows.append({"cue": match.group(0), "category": category, "context": _context(text, match)})
    return rows


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: (match.start(), match.end())) if matches else None


def _context(text: str, match: re.Match[str]) -> str:
    start = max(0, match.start() - _CONTEXT_RADIUS)
    end = min(len(text), match.end() + _CONTEXT_RADIUS)
    return text[start:end].strip()


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
