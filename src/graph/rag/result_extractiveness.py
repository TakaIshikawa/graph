"""Estimate whether retrieved result snippets are extractive enough for RAG citations."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import any_present, content_text, result_id, rounded_ratio

_QUOTE_RE = re.compile(r'"([^"]{12,})"|\'([^\']{12,})\'')
_SENTENCE_RE = re.compile(r"\b[A-Z][^.!?]{24,}[.!?]")
_SOURCE_KEYS = ("source", "source_id", "source_type", "url", "domain", "author", "publisher")


def score_result_extractiveness(results: Iterable[Any], *, min_text_chars: int = 120) -> dict[str, Any]:
    """Return per-result extractiveness scores and stable reason counts."""
    if not isinstance(min_text_chars, int) or isinstance(min_text_chars, bool) or min_text_chars < 0:
        raise ValueError("min_text_chars must be a non-negative integer")

    rows = []
    for index, result in enumerate(results):
        text = content_text(result)
        reasons = []
        if not text:
            reasons.append("missing_text")
        elif len(text) < min_text_chars:
            reasons.append("short_text")
        quote_chars = sum(len(match.group(1) or match.group(2)) for match in _QUOTE_RE.finditer(text))
        complete_sentences = len(_SENTENCE_RE.findall(text))
        has_source_metadata = any_present(result, _SOURCE_KEYS)
        if quote_chars < 40:
            reasons.append("low_quoted_span_length")
        if complete_sentences == 0 and text:
            reasons.append("no_complete_sentence")
        if not has_source_metadata:
            reasons.append("missing_source_metadata")
        score = _score(text, min_text_chars, quote_chars, complete_sentences, has_source_metadata)
        rows.append(
            {
                "result_id": result_id(result, index),
                "text_chars": len(text),
                "quoted_span_chars": quote_chars,
                "complete_sentence_count": complete_sentences,
                "has_source_metadata": has_source_metadata,
                "extractiveness_score": score,
                "reasons": reasons,
            }
        )

    reason_counts = {reason: sum(1 for row in rows if reason in row["reasons"]) for reason in sorted({reason for row in rows for reason in row["reasons"]})}
    return {
        "total_results": len(rows),
        "extractive_count": sum(1 for row in rows if row["extractiveness_score"] >= 0.7),
        "weak_count": sum(1 for row in rows if row["extractiveness_score"] < 0.5),
        "average_score": round(sum(row["extractiveness_score"] for row in rows) / len(rows), 4) if rows else 0.0,
        "reason_counts": reason_counts,
        "results": rows,
        "warnings": ["no_results"] if not rows else (["weak_extractiveness"] if any(row["extractiveness_score"] < 0.5 for row in rows) else []),
    }


def _score(text: str, min_text_chars: int, quote_chars: int, complete_sentences: int, has_source_metadata: bool) -> float:
    if not text:
        return 0.0
    text_score = min(len(text) / max(min_text_chars, 1), 1.0) * 0.35
    quote_score = min(quote_chars / 80, 1.0) * 0.25
    sentence_score = min(complete_sentences / 2, 1.0) * 0.25
    metadata_score = 0.15 if has_source_metadata else 0.0
    return round(text_score + quote_score + sentence_score + metadata_score, 4)
