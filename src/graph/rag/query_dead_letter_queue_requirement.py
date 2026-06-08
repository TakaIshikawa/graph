"""Detect dead-letter queue requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dead_letter_queue", (r"\bDLQ\b", r"\bdead[-\s]?letter\s+queues?\b")),
    ("poison_message", (r"\bpoison\s+messages?\b", r"\bpoisoned\s+messages?\b")),
    ("retry_exhaustion", (r"\bretry\s+exhaustion\b", r"\bexhausted\s+retries\b", r"\bretries\s+are\s+exhausted\b")),
    ("quarantine_queue", (r"\bquarantine\s+queues?\b", r"\bquarantine\s+messages?\b")),
    ("redrive_policy", (r"\bredrive\s+polic(?:y|ies)\b", r"\bredrive\s+from\s+(?:the\s+)?DLQ\b")),
)


def detect_query_dead_letter_queue_requirement(query: str) -> dict[str, Any]:
    """Return dead-letter queue requirement categories mentioned by a query."""
    requirements = _requirements(query)
    return {"has_dead_letter_queue_requirement": bool(requirements), "requirements": requirements}


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
