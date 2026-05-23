"""Audit whether answers fit action-oriented RAG queries."""

from __future__ import annotations

import re
from typing import Any

_ACTION_QUERY_RE = re.compile(r"\b(how|plan|steps?|implement|do|fix|choose|decide|recommend|next)\b", re.IGNORECASE)
_STEP_RE = re.compile(r"(?:^\s*(?:\d+\.|-)\s+|\b(?:first|next|then|finally)\b)", re.IGNORECASE | re.MULTILINE)
_OWNER_RE = re.compile(r"\b(owner|assigned to|team|lead|by [A-Z][a-z]+)\b")
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|today|tomorrow|next\s+week|by\s+\w+day)\b", re.IGNORECASE)
_CRITERIA_RE = re.compile(r"\b(criteria|threshold|if|when|unless|decision)\b", re.IGNORECASE)


def audit_answer_actionability_fit(query: str, answer: str) -> dict[str, Any]:
    """Return actionability fit signals for a query and answer."""
    query_requires = bool(_ACTION_QUERY_RE.search(query or ""))
    detected: list[str] = []
    if _STEP_RE.search(answer or ""):
        detected.append("steps")
    if _OWNER_RE.search(answer or ""):
        detected.append("owners")
    if _DATE_RE.search(answer or ""):
        detected.append("dates")
    if _CRITERIA_RE.search(answer or ""):
        detected.append("decision criteria")

    required = ["steps", "owners", "dates", "decision criteria"] if query_requires else []
    missing = [item for item in required if item not in detected]
    score = 1.0 if not query_requires else round((len(required) - len(missing)) / len(required), 2)
    return {
        "actionability_score": score,
        "missing_action_elements": missing,
        "detected_actions": detected,
        "query_requires_action": query_requires,
    }
