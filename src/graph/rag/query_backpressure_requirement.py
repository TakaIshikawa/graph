"""Detect backpressure requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("backpressure", (r"\bback[-\s]?pressure\b", r"\bpressure\s+control\b")),
    ("throttling_pressure", (r"\bthrottling\s+pressure\b", r"\bthrottle\s+(?:producers?|writers?|publishers?)\b")),
    ("queue_saturation", (r"\bqueue\s+saturation\b", r"\bsaturated\s+queue\b", r"\bqueue\s+(?:is\s+)?full\b")),
    ("load_shedding", (r"\bload[-\s]?shedding\b", r"\bshed\s+load\b", r"\bdrop\s+excess\s+(?:load|requests?)\b")),
    ("bounded_buffer", (r"\bbounded\s+buffers?\b", r"\bbounded\s+queues?\b", r"\bfixed[-\s]?size\s+buffers?\b")),
    ("producer_slowdown", (r"\bproducer\s+slowdown\b", r"\bslow\s+down\s+producers?\b", r"\bslow\s+publishers?\b")),
)


def detect_query_backpressure_requirement(query: str) -> dict[str, Any]:
    """Return backpressure requirement categories mentioned by a query."""
    requirements = _requirements(query)
    return {
        "has_backpressure_requirement": bool(requirements),
        "requirements": requirements,
    }


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
