"""Classify privacy-sensitive cues in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CUES: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "identity": (
        ("ssn", re.compile(r"\b(?:ssn|social\s+security\s+number)\b", re.IGNORECASE)),
        ("email address", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
        ("phone number", re.compile(r"\b(?:phone|mobile|cell)\s+number\b", re.IGNORECASE)),
        ("passport", re.compile(r"\bpassport\b", re.IGNORECASE)),
        ("driver license", re.compile(r"\bdriver'?s?\s+licen[cs]e\b", re.IGNORECASE)),
    ),
    "secret": (
        ("password", re.compile(r"\bpasswords?\b", re.IGNORECASE)),
        ("api key", re.compile(r"\bapi\s+keys?\b", re.IGNORECASE)),
        ("token", re.compile(r"\b(?:access|refresh|auth|bearer)?\s*tokens?\b", re.IGNORECASE)),
        ("secret", re.compile(r"\b(?:secret|private\s+key|credential)s?\b", re.IGNORECASE)),
    ),
    "financial": (
        ("credit card", re.compile(r"\bcredit\s+card\b", re.IGNORECASE)),
        ("bank account", re.compile(r"\bbank\s+accounts?\b", re.IGNORECASE)),
        ("routing number", re.compile(r"\brouting\s+number\b", re.IGNORECASE)),
        ("tax", re.compile(r"\b(?:tax\s+return|w-?2|1099)\b", re.IGNORECASE)),
        ("payment", re.compile(r"\b(?:payment|payroll|salary)\b", re.IGNORECASE)),
    ),
    "health": (
        ("medical", re.compile(r"\b(?:medical|clinical|health)\s+(?:record|history|data|note)s?\b", re.IGNORECASE)),
        ("diagnosis", re.compile(r"\bdiagnos(?:is|es|ed)\b", re.IGNORECASE)),
        ("prescription", re.compile(r"\bprescriptions?\b", re.IGNORECASE)),
        ("patient", re.compile(r"\bpatients?\b", re.IGNORECASE)),
        ("hipaa", re.compile(r"\bhipaa\b", re.IGNORECASE)),
    ),
    "location": (
        ("home address", re.compile(r"\bhome\s+address\b", re.IGNORECASE)),
        ("current location", re.compile(r"\bcurrent\s+location\b", re.IGNORECASE)),
        ("gps", re.compile(r"\b(?:gps|geolocation|location\s+history)\b", re.IGNORECASE)),
        ("where i live", re.compile(r"\bwhere\s+i\s+live\b", re.IGNORECASE)),
    ),
    "private_communication": (
        ("email", re.compile(r"\b(?:private\s+)?emails?\b", re.IGNORECASE)),
        ("dm", re.compile(r"\b(?:dm|direct\s+message)s?\b", re.IGNORECASE)),
        ("chat", re.compile(r"\b(?:chat|slack|discord)\s+(?:log|message|thread)s?\b", re.IGNORECASE)),
        ("text message", re.compile(r"\btext\s+messages?\b", re.IGNORECASE)),
    ),
}

_FLAG_KEYS = tuple(_CUES)


def classify_query_privacy_sensitivity(query: str) -> dict[str, Any]:
    """Return deterministic privacy-sensitivity flags for a query."""
    normalized_query = " ".join(str(query).split())
    reasons = {
        key: [label for label, pattern in cues if pattern.search(normalized_query)]
        for key, cues in _CUES.items()
    }
    flags = {key: bool(reasons[key]) for key in _FLAG_KEYS}

    return {
        "normalized_query": normalized_query,
        **flags,
        "reasons": reasons,
        "sensitivity_level": _sensitivity_level(flags),
    }


def _sensitivity_level(flags: dict[str, bool]) -> str:
    matched = {key for key, value in flags.items() if value}
    if not matched:
        return "none"
    if matched.intersection({"secret", "financial", "health"}) or len(matched) >= 3:
        return "high"
    if matched.intersection({"identity", "location", "private_communication"}):
        return "medium" if len(matched) >= 2 else "low"
    return "low"
