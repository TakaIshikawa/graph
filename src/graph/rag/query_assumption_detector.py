"""Detect premise assumptions embedded in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PREMISE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("why did", re.compile(r"\bwhy\s+did\b", re.IGNORECASE)),
    ("given that", re.compile(r"\bgiven\s+that\b", re.IGNORECASE)),
    ("since", re.compile(r"\bsince\b", re.IGNORECASE)),
    ("still", re.compile(r"\bstill\b", re.IGNORECASE)),
    ("again", re.compile(r"\bagain\b", re.IGNORECASE)),
    ("continue to", re.compile(r"\bcontinue\s+to\b", re.IGNORECASE)),
)


def detect_query_assumptions(query: str) -> dict[str, Any]:
    """Return assumption cues and verification questions for a query."""
    normalized = _inline_text(query)
    assumptions = [label for label, pattern in _PREMISE_PATTERNS if pattern.search(normalized)]
    questions = [_question_for(label) for label in assumptions]
    count = len(assumptions)
    return {
        "assumptions": assumptions,
        "assumption_count": count,
        "verification_questions": questions,
        "risk_level": "high" if count >= 3 else "medium" if count else "low",
    }


def _question_for(label: str) -> str:
    return f"Verify whether the premise signaled by '{label}' is supported before answering."


def _inline_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())
