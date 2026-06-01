"""Detect retention-period requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("retention_period", (r"\bretention\s+period\b", r"\bretain(?:ed)?\s+for\b", r"\bkeep\s+(?:data|records|logs|files)\s+for\b")),
    ("deletion_window", (r"\bdeletion\s+window\b", r"\bdelete\s+(?:data|records|logs|files)\s+(?:after|within)\b", r"\bpurg(?:e|ed|ing)\s+(?:after|within)\b")),
    ("archival_period", (r"\barchiv(?:al|e|ed|ing)\s+period\b", r"\barchive\s+(?:data|records|logs|files)\s+for\b")),
    ("purge_schedule", (r"\bpurge\s+schedule\b", r"\bscheduled\s+purge\b", r"\bpurge\s+(?:cadence|frequency)\b")),
    ("records_retention", (r"\brecords?\s+retention\b", r"\bretention\s+of\s+records?\b")),
    ("legal_retention", (r"\blegal\s+retention\b", r"\bregulatory\s+retention\b", r"\bstatutory\s+retention\b", r"\blegally\s+required\s+retention\b")),
)
_DURATION_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:hours?|hrs?|days?|weeks?|months?|quarters?|years?|yrs?)\b",
    re.I,
)
_LEGAL_RE = re.compile(r"\b(?:legal|regulatory|statutory|compliance)\s+retention\b|\blegally\s+required\s+retention\b", re.I)


def detect_query_retention_period_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    requirements = [category for category, patterns in _REQUIREMENTS if _first_match(patterns, text)]
    return {
        "has_retention_period_requirements": bool(requirements),
        "requirements": requirements,
        "explicit_duration_mentions": _duration_mentions(text) if requirements else [],
        "legal_retention_sensitive": bool(_LEGAL_RE.search(text)),
    }


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _duration_mentions(text: str) -> list[str]:
    seen: set[str] = set()
    durations: list[str] = []
    for match in _DURATION_RE.finditer(text):
        duration = " ".join(match.group(0).split())
        key = duration.casefold()
        if key not in seen:
            seen.add(key)
            durations.append(duration)
    return durations


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
