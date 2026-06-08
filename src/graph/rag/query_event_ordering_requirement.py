"""Detect event ordering requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("event_ordering", (r"\bevent\s+ordering\b", r"\bordering\s+guarantees?\b")),
    ("fifo_queue", (r"\bFIFO\s+queues?\b", r"\bfirst[-\s]?in[-\s]?first[-\s]?out\b")),
    ("sequence_number", (r"\bsequence\s+numbers?\b", r"\bmonotonic\s+sequences?\b")),
    ("out_of_order_delivery", (r"\bout[-\s]?of[-\s]?order\s+delivery\b", r"\bdelivered\s+out\s+of\s+order\b")),
    ("partition_ordering", (r"\bpartition\s+ordering\b", r"\border(?:ed|ing)\s+within\s+partitions?\b")),
)


def detect_query_event_ordering_requirement(query: str) -> dict[str, Any]:
    """Return event ordering requirement categories mentioned by a query."""
    requirements = _requirements(query)
    return {"has_event_ordering_requirement": bool(requirements), "requirements": requirements}


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
