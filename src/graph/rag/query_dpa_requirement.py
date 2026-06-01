"""Detect data-processing agreement requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_DPA = (r"\bdpa\b", r"\bdata\s+processing\s+agreement\b", r"\bprocessing\s+addendum\b", r"\bdata\s+processing\s+addendum\b")
_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("subprocessor", (r"\bsubprocessors?\b",)),
    ("processor", (r"\bprocessor\s+obligations?\b", r"\bprocessors?\b")),
    ("controller", (r"\bcontroller\s+obligations?\b", r"\bcontrollers?\b")),
    ("scc", (r"\bsccs?\b", r"\bstandard\s+contractual\s+clauses?\b")),
)


def detect_query_dpa_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    matched = _matches(text, _DPA)
    cues = [name for name, patterns in _CUES if _matches(text, patterns)]
    return {
        "requires_dpa": bool(matched),
        "matched_phrases": matched,
        "cue_categories": cues,
        "recommendations": ["retrieve data processing agreement terms"] if matched else [],
        "confidence": "high" if matched else "none",
    }


def _matches(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [match.group(0) for pattern in patterns for match in re.finditer(pattern, text, re.I)]
