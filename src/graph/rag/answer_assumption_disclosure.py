"""Audit whether a RAG answer discloses assumptions."""

from __future__ import annotations

import re
from typing import Any

_EXPLICIT_SECTION_RE = re.compile(r"(?:^|\n)\s*(?:#+\s*)?(assumptions?|key assumptions?)\s*:?", re.I)
_ASSUMPTION_PHRASES = (
    r"\bassum(?:e|ing|ption)s?\b",
    r"\bon the assumption that\b",
    r"\bprovided that\b",
    r"\bif we assume\b",
    r"\bholding .* constant\b",
)
_RISKY_QUERY_RE = re.compile(r"\b(?:estimate|forecast|predict|projection|compare|best|recommend|should|scenario)\b", re.I)


def audit_answer_assumption_disclosure(query: str, answer: str) -> dict[str, Any]:
    """Return assumption disclosure signals and missing-assumption risk."""
    normalized_query = " ".join(str(query or "").split())
    normalized_answer = " ".join(str(answer or "").split())
    explicit = bool(_EXPLICIT_SECTION_RE.search(str(answer or "")))
    inline = _phrases(normalized_answer, _ASSUMPTION_PHRASES)
    query_needs_assumptions = bool(_RISKY_QUERY_RE.search(normalized_query))
    disclosed = explicit or bool(inline)
    risk_score = 0.0
    if query_needs_assumptions:
        risk_score = 0.75 if not disclosed else 0.15
    elif normalized_answer and not disclosed:
        risk_score = 0.1
    return {
        "query_needs_assumptions": query_needs_assumptions,
        "has_assumption_disclosure": disclosed,
        "explicit_assumption_section": explicit,
        "inline_assumption_phrases": inline,
        "missing_assumption_risk": round(risk_score, 2),
        "risk_level": "high" if risk_score >= 0.7 else "medium" if risk_score >= 0.35 else "low",
    }


def _phrases(text: str, patterns: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(match.group(0).strip() for match in re.finditer(pattern, text, re.I))
    return found
