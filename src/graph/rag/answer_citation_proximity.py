"""Audit whether citations appear near answer claims."""

from __future__ import annotations

import re
from typing import Any

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_CITATION_RE = re.compile(r"(?:\[[^\[\]\n]+\]|\^\d+|https?://\S+)")


def audit_answer_citation_proximity(answer: str, *, max_distance: int = 1) -> dict[str, Any]:
    """Return citation proximity signals for answer sentences."""
    sentences = [_clean(match.group(0)) for match in _SENTENCE_RE.finditer(answer or "") if _clean(match.group(0))]
    cited_indexes = [index for index, sentence in enumerate(sentences) if _CITATION_RE.search(sentence)]
    unsupported: list[str] = []
    warnings: list[str] = []

    for index, sentence in enumerate(sentences):
        if not _claim_like(sentence):
            continue
        distance = min((abs(index - cited) for cited in cited_indexes), default=max_distance + 1)
        if distance > max_distance:
            unsupported.append(sentence)
            warnings.append(f"Claim sentence {index + 1} has no nearby citation.")

    cited_count = len(cited_indexes)
    claim_count = sum(1 for sentence in sentences if _claim_like(sentence))
    score = 1.0 if not claim_count else round((claim_count - len(unsupported)) / claim_count, 3)
    return {
        "sentence_count": len(sentences),
        "cited_sentence_count": cited_count,
        "unsupported_claim_sentences": unsupported,
        "proximity_score": score,
        "warnings": warnings,
    }


def _claim_like(sentence: str) -> bool:
    stripped = _CITATION_RE.sub("", sentence).strip()
    return len(stripped.split()) >= 2


def _clean(value: str) -> str:
    return " ".join(value.split())
