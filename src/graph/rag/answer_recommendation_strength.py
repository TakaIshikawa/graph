"""Classify recommendation language strength in RAG answers."""

from __future__ import annotations

import re
from typing import Any

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
_STRONG = ("must", "should", "need to", "required to", "recommend", "best option", "do not", "avoid")
_MODERATE = ("consider", "it is advisable", "worth", "prefer", "better to", "likely should")
_WEAK = ("may", "might", "could", "option", "one approach", "possible")
_SUPPORT_RE = re.compile(r"\[\d+\]|\(\d+\)|https?://|\b(?:because|based on|according to|evidence|study|data|limitation|unless|if|depending|caveat)\b", re.I)
_LEVEL_RANK = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}


def analyze_answer_recommendation_strength(answer: str) -> dict[str, Any]:
    """Return recommendation counts and unsupported strong recommendation sentences."""
    normalized = " ".join(str(answer or "").split())
    recommendations = []
    unsupported = []
    for sentence in _sentences(normalized):
        level, cue = _level(sentence)
        if level == "none":
            continue
        supported = level != "strong" or bool(_SUPPORT_RE.search(sentence))
        row = {"sentence": sentence, "level": level, "cue": cue, "supported": supported}
        recommendations.append(row)
        if level == "strong" and not supported:
            unsupported.append(row)
    strongest = "none"
    for row in recommendations:
        if _LEVEL_RANK[row["level"]] > _LEVEL_RANK[strongest]:
            strongest = row["level"]
    reasons = []
    if unsupported:
        reasons.append("strong_recommendations_without_evidence_or_caveats")
    if recommendations:
        reasons.append(f"{strongest}_recommendation_language_detected")
    return {
        "recommendation_count": len(recommendations),
        "strongest_level": strongest,
        "unsupported_strong_recommendations": unsupported,
        "reasons": reasons,
    }


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def _level(sentence: str) -> tuple[str, str | None]:
    folded = sentence.casefold()
    for cue in _STRONG:
        if re.search(rf"\b{re.escape(cue)}\b", folded):
            return "strong", cue
    for cue in _MODERATE:
        if re.search(rf"\b{re.escape(cue)}\b", folded):
            return "moderate", cue
    for cue in _WEAK:
        if re.search(rf"\b{re.escape(cue)}\b", folded):
            return "weak", cue
    return "none", None
