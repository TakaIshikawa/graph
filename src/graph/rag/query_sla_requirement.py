"""Detect service-level agreement evidence requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sla", (r"\bsla\b", r"\bservice-level\s+agreement\b", r"\bservice\s+level\s+agreement\b")),
    ("uptime_guarantee", (r"\buptime\s+guarantee\b", r"\buptime\s+commitment\b")),
    ("availability_target", (r"\bavailability\s+target\b", r"\bavailability\s+commitment\b")),
    ("service_credits", (r"\bservice\s+credits?\b", r"\bcredit\s+remed(?:y|ies)\b")),
    ("support_response_target", (r"\bsupport\s+response\s+targets?\b", r"\bresponse\s+(?:time|window)\b")),
    ("severity_levels", (r"\bseverity\s+levels?\b", r"\bsev\s*\d+\b", r"\bp\d+\b")),
    ("maintenance_exclusions", (r"\bmaintenance\s+exclusions?\b", r"\bexcluded\s+maintenance\b")),
)
_TARGET_RE = re.compile(
    r"\b\d+(?:\.\d+)?%|\b(?:p|sev)\s*\d+\b|\b\d+(?:\.\d+)?\s*(?:minutes?|hours?|days?)\b|\bnext\s+business\s+day\b",
    re.I,
)


def detect_query_sla_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    categories = [category for category, patterns in _CATEGORIES if _first_match(patterns, text)]
    return {
        "requires_sla": bool(categories),
        "cue_categories": categories,
        "target_values": _target_values(text) if categories else [],
    }


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _target_values(text: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for match in _TARGET_RE.finditer(text):
        value = " ".join(match.group(0).split())
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            values.append(value)
    return values


def _normalize_query(query: str) -> str:
    text = " ".join(str(query or "").split())
    if not text:
        raise ValueError("query must not be empty")
    return text
