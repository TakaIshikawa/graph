"""Score how well RAG results fit simple query intent cues."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, metadata, result_date, result_id, string, tokens, value

_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?%?(?!\w)")
_COMPARISON = ("compare", "versus", " vs ", "difference", "better", "pros", "cons")
_TIMELINE = ("timeline", "history", "when", "after", "before", "latest", "recent")
_HOW_TO = ("how to", "steps", "guide", "tutorial", "configure", "implement")
_OPINION = ("best", "recommend", "should", "worth", "opinion", "review")


def score_result_intent_fit(query: Any, results: Iterable[Any]) -> list[dict[str, Any]]:
    """Return bounded per-result fit scores for inferred query intent."""
    query_text = string(query) or ""
    intent = _intent(query_text)
    query_terms = tokens(query_text, min_length=3)
    rows = []
    for index, result in enumerate(results):
        text = content_text(result)
        lowered = f" {text.casefold()} "
        result_terms = tokens(text, min_length=3)
        matched_signals: list[str] = []
        if query_terms and result_terms & query_terms:
            matched_signals.append("query_terms")
        if _NUMBER_RE.search(text) or any(string(value(result, key)) for key in ("score", "count", "sample_size")):
            matched_signals.append("numeric_evidence")
        if result_date(result) is not None:
            matched_signals.append("date_metadata")
        if _has_any(lowered, _COMPARISON):
            matched_signals.append("comparison_language")
        if _has_any(lowered, _HOW_TO) or re.search(r"\b(step\s+\d+|first|next|then|finally)\b", lowered):
            matched_signals.append("how_to_steps")
        if _has_any(lowered, _OPINION):
            matched_signals.append("opinion_markers")
        if _source_or_author(result):
            matched_signals.append("source_attribution")

        score = _score(intent, matched_signals, query_terms, result_terms)
        warnings = []
        if not text.strip():
            warnings.append("missing_content")
        if score < 0.35:
            warnings.append("weak_intent_fit")
        rows.append(
            {
                "result_id": result_id(result, index),
                "intent": intent,
                "fit_score": score,
                "matched_signals": matched_signals,
                "warnings": warnings,
            }
        )
    return rows


def _intent(query: str) -> str:
    lowered = f" {query.casefold()} "
    if _has_any(lowered, _COMPARISON):
        return "comparison"
    if _has_any(lowered, _TIMELINE):
        return "timeline"
    if _has_any(lowered, _HOW_TO):
        return "how_to"
    if _has_any(lowered, _OPINION):
        return "opinion"
    return "fact_lookup"


def _score(intent: str, signals: list[str], query_terms: set[str], result_terms: set[str]) -> float:
    overlap = len(query_terms & result_terms) / max(1, len(query_terms))
    score = min(0.35, overlap * 0.35)
    weights = {
        "comparison": {"comparison_language": 0.35, "numeric_evidence": 0.15, "source_attribution": 0.1},
        "timeline": {"date_metadata": 0.35, "numeric_evidence": 0.1, "source_attribution": 0.1},
        "how_to": {"how_to_steps": 0.35, "source_attribution": 0.1},
        "opinion": {"opinion_markers": 0.25, "comparison_language": 0.15, "source_attribution": 0.1},
        "fact_lookup": {"numeric_evidence": 0.15, "source_attribution": 0.15, "date_metadata": 0.1},
    }[intent]
    score += sum(weight for signal, weight in weights.items() if signal in signals)
    return round(min(1.0, score), 3)


def _has_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


def _source_or_author(result: Any) -> bool:
    return any(string(value(result, key)) for key in ("author", "source", "source_name", "domain")) or bool(metadata(result))
