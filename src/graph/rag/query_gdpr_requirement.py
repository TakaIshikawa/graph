"""Detect GDPR requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_GDPR = (r"\bgdpr\b", r"\bgeneral\s+data\s+protection\s+regulation\b", r"\beu\s+personal\s+data\b")
_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rights", (r"\bdata\s+subject\s+rights?\b", r"\bdsr\b", r"\bright\s+to\s+erasure\b", r"\berasure\b")),
    ("lawful_basis", (r"\blawful\s+basis\b", r"\blegal\s+basis\b")),
    ("controller_processor", (r"\bcontrollers?\b", r"\bprocessors?\b")),
    ("transfer", (r"\bsccs?\b", r"\bstandard\s+contractual\s+clauses?\b", r"\binternational\s+transfer\b")),
    ("dpia", (r"\bdpia\b", r"\bdata\s+protection\s+impact\s+assessment\b")),
)


def detect_query_gdpr_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    matched = _matches(text, _GDPR)
    cues = [name for name, patterns in _CUES if _matches(text, patterns)]
    return {
        "requires_gdpr": bool(matched),
        "matched_phrases": matched,
        "cue_categories": cues,
        "recommendations": ["retrieve GDPR compliance evidence"] if matched else [],
        "confidence": "high" if matched else "none",
    }


def _matches(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [match.group(0) for pattern in patterns for match in re.finditer(pattern, text, re.I)]
