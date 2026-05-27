"""Analyze wasted or excessive context window usage."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, number, result_id, string, value

_TOKEN_KEYS = ("token_count", "tokens", "estimated_tokens")
_SCORE_KEYS = ("score", "relevance_score", "rank_score")


def analyze_context_window_waste(chunks: Iterable[Any], *, max_tokens: int) -> dict[str, Any]:
    """Estimate context utilization and flag chunks that add little value."""
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
        raise ValueError("max_tokens must be a non-negative integer")
    chunk_list = list(chunks or [])
    chunk_rows = [_chunk_row(chunk, index) for index, chunk in enumerate(chunk_list)]
    used_tokens = sum(row["token_count"] for row in chunk_rows)
    unused_tokens = max(max_tokens - used_tokens, 0)
    over_budget_tokens = max(used_tokens - max_tokens, 0)
    utilization_ratio = round(used_tokens / max_tokens, 3) if max_tokens else 0.0
    low_value_chunks = [row for row in chunk_rows if row["is_low_value"]]
    return {
        "used_tokens": used_tokens,
        "unused_tokens": unused_tokens,
        "over_budget_tokens": over_budget_tokens,
        "utilization_ratio": utilization_ratio,
        "low_value_chunks": low_value_chunks,
        "recommendations": _recommendations(unused_tokens, over_budget_tokens, low_value_chunks),
    }


def _chunk_row(chunk: Any, index: int) -> dict[str, Any]:
    text = content_text(chunk)
    tokens = _token_count(chunk, text)
    score = _score(chunk)
    reasons = []
    if score == 0.0:
        reasons.append("zero_score")
    if tokens <= 3 or len(text) < 20:
        reasons.append("very_small_text_contribution")
    return {
        "result_id": result_id(chunk, index),
        "token_count": tokens,
        "score": score,
        "reasons": reasons,
        "is_low_value": bool(reasons),
    }


def _token_count(chunk: Any, text: str) -> int:
    for key in _TOKEN_KEYS:
        explicit = number(value(chunk, key))
        if explicit is not None:
            return max(int(explicit), 0)
    normalized = string(text) or ""
    return math.ceil(len(normalized) / 4) if normalized else 0


def _score(chunk: Any) -> float:
    for key in _SCORE_KEYS:
        score = number(value(chunk, key))
        if score is not None:
            return max(score, 0.0)
    return 0.0


def _recommendations(unused_tokens: int, over_budget_tokens: int, low_value_chunks: list[dict[str, Any]]) -> list[str]:
    recommendations = []
    if over_budget_tokens:
        recommendations.append("trim_context_to_fit_window")
    if unused_tokens:
        recommendations.append("use_remaining_budget_for_higher_recall_evidence")
    if low_value_chunks:
        recommendations.append("remove_or_replace_low_value_chunks")
    return recommendations
