"""Detect circuit breaker requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("circuit_breaker", (r"\bcircuit\s+breakers?\b",)),
    ("breaker_state", (r"\bopen\s+state\b", r"\bhalf[-\s]?open\b", r"\bclosed\s+state\b")),
    ("failure_threshold", (r"\bfailure\s+threshold\b", r"\berror\s+threshold\b", r"\btrip\s+threshold\b")),
    ("fallback_path", (r"\bfallback\s+(?:path|route|response|behavior)\b", r"\bfail\s+over\s+to\b")),
    ("dependency_isolation", (r"\bdependency\s+isolation\b", r"\bisolate\s+(?:failing\s+)?dependenc(?:y|ies)\b")),
)


def detect_query_circuit_breaker_requirement(query: str) -> dict[str, Any]:
    """Return circuit breaker requirement categories mentioned by a query."""
    requirements = _requirements(query)
    return {"has_circuit_breaker_requirement": bool(requirements), "requirements": requirements}


def _requirements(query: str) -> list[dict[str, str]]:
    text = " ".join(str(query or "").split())
    rows: list[dict[str, str]] = []
    for category, patterns in _CATEGORIES:
        match = _first_match(patterns, text)
        if match:
            rows.append({"category": category, "matched_text": match.group(0)})
    return sorted(rows, key=lambda row: row["category"])


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None
