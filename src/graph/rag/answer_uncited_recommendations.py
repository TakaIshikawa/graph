"""Detect recommendation sentences that lack citations."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import string

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_RECOMMENDATION_RE = re.compile(r"\b(?:should|recommend(?:ed|s|ing)?|best|avoid|choose|consider)\b", re.I)
_CITATION_RE = re.compile(r"\[[A-Za-z0-9][A-Za-z0-9_.:-]*\]|https?://\S+", re.I)


def audit_answer_uncited_recommendations(answer: Any) -> dict[str, Any]:
    """Return recommendation citation coverage for an answer."""
    recommendations = [sentence for sentence in _sentences(answer) if _RECOMMENDATION_RE.search(sentence)]
    uncited = [sentence for sentence in recommendations if not _CITATION_RE.search(sentence)]
    cited_count = len(recommendations) - len(uncited)

    return {
        "recommendation_count": len(recommendations),
        "cited_recommendation_count": cited_count,
        "uncited_recommendation_count": len(uncited),
        "uncited_ratio": 0.0 if not recommendations else round(len(uncited) / len(recommendations), 4),
        "findings": [{"type": "uncited_recommendation", "snippet": sentence} for sentence in uncited],
    }


def _sentences(answer: Any) -> list[str]:
    return [" ".join(match.group(0).strip().split()) for match in _SENTENCE_RE.finditer(string(answer) or "") if match.group(0).strip()]
