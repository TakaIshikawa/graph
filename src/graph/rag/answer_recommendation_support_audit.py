"""Audit whether answer recommendations are supported by evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, ordered_terms, string

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_RECOMMENDATION_RE = re.compile(r"\b(?:should|recommend(?:ed|s|ing)?|best|avoid|prefer|choose|use|do not|don't)\b", re.I)


def audit_answer_recommendation_support(answer: object, evidence: Iterable[Any] | Any) -> dict[str, Any]:
    """Return support counts for deduplicated recommendation sentences."""
    evidence_texts = _evidence_texts(evidence)
    recommendations = _recommendation_sentences(answer)
    unsupported = []
    supported_count = 0

    for sentence in recommendations:
        support = _supporting_evidence(sentence, evidence_texts)
        if support:
            supported_count += 1
        else:
            unsupported.append(sentence)

    count = len(recommendations)
    unsupported_count = len(unsupported)
    if unsupported_count == 0:
        severity = "none"
    elif supported_count == 0:
        severity = "high"
    elif unsupported_count / count >= 0.5:
        severity = "medium"
    else:
        severity = "low"

    return {
        "recommendation_count": count,
        "supported_count": supported_count,
        "unsupported_recommendations": unsupported,
        "severity": severity,
    }


def _recommendation_sentences(answer: object) -> list[str]:
    text = string(answer) or ""
    seen: set[str] = set()
    sentences = []
    for match in _SENTENCE_RE.finditer(text):
        sentence = " ".join(match.group(0).strip().split())
        key = sentence.casefold()
        if sentence and key not in seen and _RECOMMENDATION_RE.search(sentence):
            seen.add(key)
            sentences.append(sentence)
    return sentences


def _evidence_texts(evidence: Iterable[Any] | Any) -> list[str]:
    if evidence is None:
        return []
    if isinstance(evidence, str):
        items = [evidence]
    else:
        try:
            items = list(evidence)
        except TypeError:
            items = [evidence]
    texts = []
    for item in items:
        text = item if isinstance(item, str) else content_text(item) or string(item) or ""
        if text:
            texts.append(text.casefold())
    return texts


def _supporting_evidence(sentence: str, evidence_texts: list[str]) -> bool:
    terms = [term for term in ordered_terms(sentence, min_length=4) if term not in {"should", "recommend", "recommended", "best", "avoid", "prefer", "choose"}]
    if not terms:
        return False
    for text in evidence_texts:
        overlap = sum(1 for term in terms if term in text)
        if overlap >= min(2, len(terms)):
            return True
    return False
