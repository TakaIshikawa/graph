"""Audit numeric answer claims for missing caveat qualifiers."""

from __future__ import annotations

import re
from typing import Any

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_PERCENT_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)?\s?%")
_CURRENCY_RE = re.compile(r"[$€£]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\b\d+(?:\.\d+)?\s?(?:USD|EUR|GBP)\b", re.I)
_AVERAGE_RE = re.compile(
    r"\b(?:average|avg\.?|mean|median)\b(?:\s+\w+){0,3}?\s+(?:was|is|of|:)?\s*(?:[$€£]\s?)?\d+(?:,\d{3})*(?:\.\d+)?%?\b",
    re.I,
)
_RANGE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|[$€£])?\s?(?:-|to|through|between)\s?(?:[$€£]\s?)?\d+(?:\.\d+)?\s?%?\b", re.I)
_COUNT_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s+(?:users?|people|patients?|cases?|items?|records?|responses?|studies|companies|units|events|transactions)\b",
    re.I,
)
_DENOMINATOR_RE = re.compile(r"\b(?:out of|of|among|per|from|sample of|n\s*=|denominator|base)\s+[\w,]+", re.I)
_TIMEFRAME_RE = re.compile(
    r"\b(?:in|during|over|from|between|since|as of|through|by)\s+(?:Q[1-4]\s+)?(?:19|20)\d{2}\b|\b(?:today|yesterday|last|past|previous|current|monthly|quarterly|annual|yearly|week|month|year)\b",
    re.I,
)
_SOURCE_RE = re.compile(r"\b(?:according to|reported by|cited by|source:|survey|study|dataset|report|estimate)\b|(?:\[\d+\]|\(\d+\)|https?://)", re.I)


def analyze_answer_numeric_caveats(answer: str) -> dict[str, Any]:
    """Return numeric claims and caveat gaps for denominator, timeframe, and source."""
    normalized = " ".join(str(answer or "").split())
    claims = _numeric_claims(normalized)
    gaps = {"missing_denominator": [], "missing_timeframe": [], "missing_source": []}
    for claim in claims:
        for qualifier in ("denominator", "timeframe", "source"):
            if not claim[f"has_{qualifier}"]:
                gaps[f"missing_{qualifier}"].append(claim["text"])
    gap_count = sum(len(values) for values in gaps.values())
    if gap_count == 0:
        risk = "low"
    elif gap_count <= 2:
        risk = "medium"
    else:
        risk = "high"
    return {
        "normalized_answer": normalized,
        "numeric_claims": claims,
        "caveat_gaps": gaps,
        "risk_level": risk,
    }


def _numeric_claims(text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    seen: list[tuple[int, int]] = []
    patterns = (
        ("average", _AVERAGE_RE),
        ("range", _RANGE_RE),
        ("percent", _PERCENT_RE),
        ("currency", _CURRENCY_RE),
        ("count", _COUNT_RE),
    )
    for sentence_match in _SENTENCE_RE.finditer(text):
        sentence = sentence_match.group(0).strip()
        for claim_type, pattern in patterns:
            for match in pattern.finditer(sentence):
                span = (sentence_match.start() + match.start(), sentence_match.start() + match.end())
                if any(_overlaps(span, existing) for existing in seen):
                    continue
                seen.append(span)
                claims.append(_claim(claim_type, match.group(0), sentence))
    return claims


def _claim(claim_type: str, matched: str, sentence: str) -> dict[str, Any]:
    has_denominator = bool(_DENOMINATOR_RE.search(sentence))
    has_timeframe = bool(_TIMEFRAME_RE.search(sentence) or _YEAR_RE.search(sentence))
    has_source = bool(_SOURCE_RE.search(sentence))
    missing = [
        name
        for name, present in (
            ("denominator", has_denominator),
            ("timeframe", has_timeframe),
            ("source", has_source),
        )
        if not present
    ]
    return {
        "claim_type": claim_type,
        "text": matched.strip(),
        "sentence": sentence,
        "has_denominator": has_denominator,
        "has_timeframe": has_timeframe,
        "has_source": has_source,
        "missing_qualifiers": missing,
    }


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]
