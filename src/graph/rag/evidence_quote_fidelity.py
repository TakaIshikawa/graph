"""Score whether quoted evidence appears faithfully in source text."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_id, string, tokens, value

_QUOTE_KEYS = ("quote", "quoted_text", "evidence_quote", "snippet", "excerpt")
_SOURCE_KEYS = ("source_text", "content", "text", "body", "document")
_SPACE_RE = re.compile(r"\s+")


def score_evidence_quote_fidelity(results: Iterable[Any]) -> list[dict[str, Any]]:
    """Return quote/source fidelity rows for retrieved evidence records."""
    try:
        rows = list(results or [])
    except TypeError:
        rows = []

    return [_score_row(result, index) for index, result in enumerate(rows)]


def _score_row(result: Any, index: int) -> dict[str, Any]:
    quote = _first_text(result, _QUOTE_KEYS)
    source = _first_text(result, _SOURCE_KEYS, exclude=quote)
    fidelity, score, reason = _classify(quote, source)
    return {
        "result_id": result_id(result, index),
        "fidelity": fidelity,
        "score": score,
        "quote": quote,
        "reason": reason,
    }


def _first_text(result: Any, keys: tuple[str, ...], *, exclude: str | None = None) -> str | None:
    for key in keys:
        text = string(value(result, key))
        if text is not None and text != exclude:
            return text
    return None


def _classify(quote: str | None, source: str | None) -> tuple[str, float, str]:
    if quote is None:
        return "missing", 0.0, "missing quote"
    if source is None:
        return "missing", 0.0, "missing source text"
    if quote in source:
        return "exact", 1.0, "quote is an exact substring of source text"

    normalized_quote = _normalize(quote)
    normalized_source = _normalize(source)
    if normalized_quote and normalized_quote in normalized_source:
        return "normalized", 0.85, "quote matches after whitespace and case normalization"

    overlap = tokens(quote).intersection(tokens(source))
    quote_tokens = tokens(quote)
    if quote_tokens and overlap:
        ratio = len(overlap) / len(quote_tokens)
        return "partial", round(max(0.2, min(0.65, ratio * 0.65)), 3), "quote partially overlaps source tokens"

    return "missing", 0.0, "quote not found in source text"


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", text.casefold()).strip()
