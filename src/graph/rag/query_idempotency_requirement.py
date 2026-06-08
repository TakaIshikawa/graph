"""Detect idempotency requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("idempotency_key", "high", (r"\bidempotency[-\s]?keys?\b", r"\bidempotent\s+keys?\b")),
    ("safe_retry", "high", (r"\bsafe\s+retr(?:y|ies)\b", r"\bretr(?:y|ies)\s+safely\b", r"\bidempotent\s+retr(?:y|ies)\b")),
    (
        "duplicate_request",
        "high",
        (r"\bduplicate\s+requests?\b", r"\bdeduplicat(?:e|ion)\s+requests?\b", r"\brequest\s+deduplication\b"),
    ),
    ("replayed_submission", "medium", (r"\breplayed\s+submissions?\b", r"\breplay\s+(?:a\s+)?submissions?\b")),
)


def detect_query_idempotency_requirement(query: str) -> dict[str, Any]:
    """Return idempotency requirements mentioned by a query."""
    text = _normalize_query(query)
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_idempotency_requirement": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
