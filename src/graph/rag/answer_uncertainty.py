"""Annotate uncertainty cues in drafted RAG answers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, tokens

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_CITATION_RE = re.compile(r"(?:\[\d+\]|\([A-Za-z][^)]+,\s*(?:19|20)\d{2}\))")
_NUMBER_OR_DATE_RE = re.compile(r"\b(?:\d+(?:[.,]\d+)*(?:%| percent)?|(?:19|20)\d{2})\b", re.IGNORECASE)
_HEDGES = (
    "may",
    "might",
    "could",
    "appears",
    "likely",
    "possibly",
    "unclear",
    "suggests",
)
_STRONG_MODAL = {"must", "will", "always", "never"}
_WEAK_MODAL = {"may", "might", "could", "possibly"}


def _sentences(answer: Any) -> list[str]:
    text = "" if answer is None else str(answer)
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def _has_support(sentence: str, result_terms: list[set[str]]) -> bool:
    terms = tokens(sentence)
    if not terms:
        return False
    return any(len(terms & row_terms) / len(terms) >= 0.5 for row_terms in result_terms if row_terms)


def annotate_answer_uncertainty(answer: Any, results: Iterable[Any] | None = None) -> dict[str, Any]:
    """Return sentence-level uncertainty annotations and a bounded score."""
    try:
        result_rows = list(results or [])
    except TypeError:
        result_rows = []
    result_terms = [tokens(content_text(result)) for result in result_rows]

    annotations: list[dict[str, Any]] = []
    total_weight = 0

    for sentence in _sentences(answer):
        lower_terms = tokens(sentence, min_length=2)
        cues: list[str] = []

        if any(hedge in lower_terms for hedge in _HEDGES):
            cues.append("hedging_language")
        cited = bool(_CITATION_RE.search(sentence))
        supported = cited or _has_support(sentence, result_terms)

        if _NUMBER_OR_DATE_RE.search(sentence) and not supported:
            cues.append("unsupported_numeric_or_date_claim")
        if not cited and len(tokens(sentence)) >= 4 and not supported:
            cues.append("uncited_factual_sentence")
        if lower_terms & _STRONG_MODAL and lower_terms & _WEAK_MODAL:
            cues.append("conflicting_modality")

        if not cues:
            continue

        severity = "low"
        weight = 1
        if "unsupported_numeric_or_date_claim" in cues or "conflicting_modality" in cues:
            severity = "high"
            weight = 3
        elif "uncited_factual_sentence" in cues:
            severity = "medium"
            weight = 2
        total_weight += weight
        annotations.append(
            {
                "sentence": sentence,
                "cues": cues,
                "severity": severity,
                "suggested_action": "Add citation or corroborating evidence for this sentence.",
            }
        )

    sentence_count = max(len(_sentences(answer)), 1)
    uncertainty_score = min(1.0, round(total_weight / (sentence_count * 3), 4))
    return {
        "annotations": annotations,
        "uncertainty_score": uncertainty_score,
        "counts": {
            "sentence_count": len(_sentences(answer)),
            "annotation_count": len(annotations),
            "result_count": len(result_rows),
        },
    }
