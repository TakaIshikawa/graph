"""Detect risk tolerance preferences in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_TOLERANCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("conservative", (r"\bconservative\b", r"\bcautious\b", r"\blow[- ]risk\b", r"\brisk[- ]averse\b", r"\bsafest\b")),
    ("aggressive", (r"\baggressive\b", r"\bhigh[- ]risk\b", r"\brisk[- ]seeking\b", r"\bexperimental\b", r"\bbleeding edge\b")),
)
_SAFETY_CRITICAL = (
    r"\bsafety[- ]critical\b",
    r"\blife[- ]critical\b",
    r"\bdo no harm\b",
    r"\bpatient safety\b",
    r"\bmission[- ]critical\b",
    r"\bregulatory risk\b",
)


def detect_query_risk_tolerance_requirement(query: str) -> dict[str, Any]:
    """Return risk preference language without treating generic uncertainty as tolerance."""
    normalized = " ".join(str(query or "").split())
    requirements = []
    for tolerance, patterns in _TOLERANCES:
        phrases = _phrases(normalized, patterns)
        if phrases:
            requirements.append({"tolerance": tolerance, "matched_phrases": phrases, "confidence": 0.84})
    safety = _phrases(normalized, _SAFETY_CRITICAL)
    return {
        "query": normalized,
        "requires_risk_tolerance": bool(requirements),
        "risk_tolerances": [row["tolerance"] for row in requirements],
        "requirements": requirements,
        "matched_phrases": [phrase for row in requirements for phrase in row["matched_phrases"]],
        "safety_critical": bool(safety),
        "safety_critical_phrases": safety,
    }


def _phrases(query: str, patterns: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, query, re.I):
            phrase = match.group(0).strip()
            key = phrase.casefold()
            if key not in seen:
                seen.add(key)
                found.append(phrase)
    return found
