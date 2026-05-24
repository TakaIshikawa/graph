"""Audit quantitative answer claims for missing denominators."""

from __future__ import annotations

import re
from typing import Any

_CLAIM_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*(?:percent|percentage points|per\s+\w+)\b|"
    r"\b\d+(?:\.\d+)?\s*[:/]\s*\d+(?:\.\d+)?\b|\b(?:average|mean|rate|share|ratio)\b)",
    re.I,
)
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_DENOMINATOR_RE = re.compile(r"\b(?:of|out of|among|per|denominator|base|n\s*=|sample of)\b", re.I)
_POPULATION_RE = re.compile(r"\b(?:patients?|users?|customers?|workers?|students?|respondents?|households?|companies?|population|sample)\b", re.I)
_TIMEFRAME_RE = re.compile(r"\b(?:in|during|between|from|since|through|over|year|month|week|quarter|day|202\d|201\d)\b", re.I)
_UNIT_RE = re.compile(r"(?:%|\b(?:percent|points?|per\s+\w+|dollars?|usd|hours?|days?|people|users?|cases?|units?)\b)", re.I)


def audit_answer_denominators(answer: str) -> dict[str, Any]:
    """Return quantitative claims with missing denominator-related fields."""
    text = str(answer or "").strip()
    claims = []
    for sentence_index, sentence in enumerate(_sentences(text)):
        for match in _CLAIM_RE.finditer(sentence):
            window = sentence[max(0, match.start() - 90) : match.end() + 90]
            missing = _missing_fields(window)
            claims.append(
                {
                    "sentence_index": sentence_index,
                    "span": match.group(0),
                    "claim_text": sentence[:220],
                    "missing_fields": missing,
                    "status": "ambiguous" if missing else "supported",
                }
            )
    reason_counts = {
        reason: sum(1 for claim in claims if reason in claim["missing_fields"])
        for reason in ("missing_denominator", "missing_population", "missing_timeframe", "missing_unit")
    }
    return {
        "total_claims": len(claims),
        "ambiguous_claims": sum(1 for claim in claims if claim["status"] == "ambiguous"),
        "claims": claims,
        "reason_counts": {key: value for key, value in reason_counts.items() if value},
        "warnings": ["ambiguous_quantitative_claims"] if any(claim["status"] == "ambiguous" for claim in claims) else [],
    }


def _missing_fields(text: str) -> list[str]:
    checks = (
        ("missing_denominator", _DENOMINATOR_RE),
        ("missing_population", _POPULATION_RE),
        ("missing_timeframe", _TIMEFRAME_RE),
        ("missing_unit", _UNIT_RE),
    )
    return [name for name, pattern in checks if not pattern.search(text)]


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]
