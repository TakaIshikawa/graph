"""Detect time horizon cues in RAG queries."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import string

_RELATIVE_RE = re.compile(r"\b(?:last|past)\s+(\d+)\s+(days|weeks|months|years)\b", re.I)
_SINCE_RE = re.compile(r"\bsince\s+((?:19|20)\d{2})\b", re.I)
_QUARTER_RE = re.compile(r"\bQ([1-4])\s*((?:19|20)\d{2})\b", re.I)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_FRESH_RE = re.compile(r"\b(?:current|latest|recent|today|now|up-to-date)\b", re.I)
_HISTORICAL_RE = re.compile(r"\b(?:historical|history|over\s+time|past\s+decade)\b", re.I)


def detect_query_time_horizon(query: Any) -> dict[str, Any]:
    """Return a compact time horizon classification for a query."""
    text = string(query) or ""
    if match := _RELATIVE_RE.search(text):
        cue = match.group(0)
        return _result("relative", f"{match.group(1)} {match.group(2).casefold()}", [cue], True, 0.9)
    if re.search(r"\bthis\s+year\b", text, re.I):
        return _result("relative", "this year", ["this year"], True, 0.8)
    if match := _SINCE_RE.search(text):
        return _result("absolute", f"since {match.group(1)}", [match.group(0)], True, 0.85)
    if match := _QUARTER_RE.search(text):
        return _result("absolute", f"{match.group(2)}-Q{match.group(1)}", [match.group(0)], False, 0.9)
    if match := _FRESH_RE.search(text):
        return _result("freshness", match.group(0).casefold(), [match.group(0)], True, 0.75)
    if match := _HISTORICAL_RE.search(text):
        return _result("historical", match.group(0).casefold(), [match.group(0)], False, 0.75)
    if match := _YEAR_RE.search(text):
        return _result("absolute", match.group(1), [match.group(1)], False, 0.7)
    return _result("absent", None, [], False, 0.0)


def _result(horizon_type: str, normalized_value: str | None, cues: list[str], fresh: bool, confidence: float) -> dict[str, Any]:
    return {
        "horizon_type": horizon_type,
        "normalized_value": normalized_value,
        "cues": cues,
        "requires_fresh_sources": fresh,
        "confidence": confidence,
    }
