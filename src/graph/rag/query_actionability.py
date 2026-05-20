"""Classify actionable-output intent in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CUES: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "checklist": (
        ("checklist", re.compile(r"\bcheck\s*list\b|\bchecklist\b", re.IGNORECASE)),
        ("todo list", re.compile(r"\b(?:to-?do|task)\s+list\b", re.IGNORECASE)),
        ("items to verify", re.compile(r"\bitems?\s+to\s+(?:verify|review|check)\b", re.IGNORECASE)),
    ),
    "stepwise": (
        ("step by step", re.compile(r"\bstep[-\s]+by[-\s]+step\b", re.IGNORECASE)),
        ("steps", re.compile(r"(?<!-)\bsteps?\b(?!-)", re.IGNORECASE)),
        ("plan", re.compile(r"\b(?:plan|roadmap|sequence)\b", re.IGNORECASE)),
        ("workflow", re.compile(r"\bworkflow\b", re.IGNORECASE)),
    ),
    "recommendation": (
        ("recommend", re.compile(r"\brecommend(?:ation|ed|s)?\b", re.IGNORECASE)),
        ("should i", re.compile(r"\bshould\s+(?:i|we)\b", re.IGNORECASE)),
        ("choose", re.compile(r"\b(?:choose|pick|select|decide)\b", re.IGNORECASE)),
        ("best option", re.compile(r"\bbest\s+(?:option|choice|approach)\b", re.IGNORECASE)),
    ),
    "implementation": (
        ("implement", re.compile(r"\bimplement(?:ation|ing)?\b", re.IGNORECASE)),
        ("build", re.compile(r"\b(?:build|create|write|code)\b", re.IGNORECASE)),
        ("configure", re.compile(r"\b(?:configure|set\s+up|install|deploy)\b", re.IGNORECASE)),
        ("instructions", re.compile(r"\binstructions?\b", re.IGNORECASE)),
    ),
    "troubleshooting": (
        ("troubleshoot", re.compile(r"\btroubleshoot(?:ing)?\b", re.IGNORECASE)),
        ("debug", re.compile(r"\bdebug(?:ging)?\b", re.IGNORECASE)),
        ("fix", re.compile(r"\bfix(?:ing)?\b", re.IGNORECASE)),
        ("error", re.compile(r"\b(?:error|failure|failing|broken|issue)\b", re.IGNORECASE)),
    ),
}

_PRIORITY = ("troubleshooting", "implementation", "recommendation", "stepwise", "checklist")


def classify_query_actionability(query: str) -> dict[str, Any]:
    """Return stable flags for actionable answer formats requested by a query."""
    normalized_query = " ".join(str(query).split())
    reasons = {
        key: [label for label, pattern in cues if pattern.search(normalized_query)]
        for key, cues in _CUES.items()
    }
    flags = {key: bool(reasons[key]) for key in _CUES}
    primary = _primary_action_type(flags)

    return {
        "normalized_query": normalized_query,
        **flags,
        "primary_action_type": primary,
        "actionability_level": _actionability_level(flags),
        "reasons": reasons,
    }


def _primary_action_type(flags: dict[str, bool]) -> str:
    for action_type in _PRIORITY:
        if flags[action_type]:
            return action_type
    return "none"


def _actionability_level(flags: dict[str, bool]) -> str:
    count = sum(1 for value in flags.values() if value)
    if count == 0:
        return "none"
    if flags["troubleshooting"] or flags["implementation"] or count >= 3:
        return "high"
    if flags["recommendation"] or flags["stepwise"]:
        return "medium"
    return "low"
