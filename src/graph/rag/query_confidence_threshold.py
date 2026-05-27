"""Detect confidence and uncertainty handling requirements in queries."""

from __future__ import annotations

import re
from typing import Any

_TERMS = {
    "high_confidence": r"\bhigh confidence\b|\bvery confident\b",
    "sure": r"\bsure\b|\bcertain\b|\bcertainty\b",
    "evidence_strength": r"\bevidence strength\b|\bstrong evidence\b",
    "flag_uncertainty": r"\bflag uncertainty\b|\bnote uncertainty\b|\bcall out uncertainty\b",
}
_PERCENT_RE = re.compile(r"\b(?:at least|above|over|>=)?\s*(\d+(?:\.\d+)?)\s*%", re.I)


def detect_query_confidence_thresholds(query: str) -> dict[str, Any]:
    text = str(query or "")
    terms = [term for term, pattern in _TERMS.items() if re.search(pattern, text, re.I)]
    thresholds = [match.group(1) + "%" for match in _PERCENT_RE.finditer(text)]
    phrases = []
    phrases.extend(terms)
    phrases.extend(thresholds)
    return {
        "requires_confidence_handling": bool(terms or thresholds),
        "threshold_values": thresholds,
        "confidence_terms": terms,
        "matched_phrases": phrases,
    }
