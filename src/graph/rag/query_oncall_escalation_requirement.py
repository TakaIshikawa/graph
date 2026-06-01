"""Detect on-call and escalation requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENT = (
    r"\bon[-\s]?call\b",
    r"\bpager\b",
    r"\bescalation\s+(?:policy|matrix)\b",
    r"\bsupport\s+escalation\b",
    r"\bafter[-\s]?hours\s+coverage\b",
    r"\bseverity\s+levels?\b",
)
_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("coverage", (r"\bafter[-\s]?hours\s+coverage\b", r"\bcoverage\b", r"\bon[-\s]?call\b")),
    ("severity", (r"\bseverity\s+levels?\b", r"\bp[12]\b", r"\bsev(?:erity)?\s*[12]\b")),
    ("owner", (r"\bresponse\s+owners?\b",)),
    ("pager", (r"\bpager\b",)),
)


def detect_query_oncall_escalation_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    matched = _matches(text, _REQUIREMENT)
    cues = [name for name, patterns in _CUES if _matches(text, patterns)]
    return {
        "requires_oncall_escalation": bool(matched),
        "matched_phrases": matched,
        "cue_categories": cues,
        "confidence": "high" if matched else "none",
    }


def _matches(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [match.group(0) for pattern in patterns for match in re.finditer(pattern, text, re.I)]
