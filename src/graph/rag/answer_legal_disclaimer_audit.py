"""Audit legal-sensitive RAG answers for disclaimer coverage."""

from __future__ import annotations

import re
from typing import Any

_LEGAL_RE = re.compile(r"\b(?:legal|law|lawsuit|sue|court|contract|liability|statute|regulation|compliance|tenant|employment|copyright|trademark|patent|divorce|custody|criminal|immigration)\b", re.I)
_JURISDICTION_RE = re.compile(r"\b(?:jurisdiction|state\s+law|local\s+law|country|province|var(?:y|ies)\s+by\s+(?:state|jurisdiction|country)|depending\s+on\s+where)\b", re.I)
_DISCLAIMER_RE = re.compile(r"\b(?:not\s+legal\s+advice|informational\s+purposes|does\s+not\s+create\s+(?:an\s+)?attorney[-\s]client|attorney[-\s]client\s+relationship|lawyer[-\s]client\s+relationship)\b", re.I)
_COUNSEL_RE = re.compile(r"\b(?:consult|speak\s+with|contact|hire|ask)\s+(?:a\s+)?(?:qualified\s+)?(?:attorney|lawyer|legal\s+counsel|solicitor)\b", re.I)
_SPECIFIC_ADVICE_RE = re.compile(r"\b(?:you\s+should\s+(?:sue|file|withhold|terminate|sign|ignore)|you\s+must\s+(?:sue|file|withhold|terminate|sign|ignore)|definitely\s+(?:sue|file|withhold|terminate|sign)|guaranteed\s+to\s+win)\b", re.I)


def audit_answer_legal_disclaimer(answer: str, query: str | None = None) -> dict[str, Any]:
    """Return legal sensitivity and disclaimer audit flags for an answer."""
    answer_text = _normalize_text(answer)
    query_text = _normalize_text(query or "")
    combined = f"{query_text} {answer_text}".strip()
    sensitive_hits = _hits(combined, _LEGAL_RE)
    advice_flags = _hits(answer_text, _SPECIFIC_ADVICE_RE)
    has_jurisdiction = bool(_JURISDICTION_RE.search(answer_text))
    has_disclaimer = bool(_DISCLAIMER_RE.search(answer_text))
    has_counsel = bool(_COUNSEL_RE.search(answer_text))
    score = min(1.0, round(0.2 * len(sensitive_hits) + (0.35 if advice_flags else 0.0), 2))
    recommendations = []
    if score and not has_jurisdiction:
        recommendations.append("add_jurisdiction_caveat")
    if score and not has_disclaimer:
        recommendations.append("add_professional_disclaimer")
    if score and not has_counsel:
        recommendations.append("recommend_consulting_qualified_counsel")
    if advice_flags:
        recommendations.append("soften_over_specific_legal_advice")
    return {
        "legal_sensitivity_score": score,
        "has_jurisdiction_caveat": has_jurisdiction,
        "has_professional_disclaimer": has_disclaimer and has_counsel,
        "has_specific_legal_advice_flags": advice_flags,
        "recommendations": recommendations,
    }


def _hits(text: str, pattern: re.Pattern[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for match in pattern.finditer(text):
        value = match.group(0).casefold()
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())
