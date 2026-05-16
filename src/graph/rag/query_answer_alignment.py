"""Check whether a drafted answer aligns with a query and retrieved context."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_STOPWORDS = {
    "about", "after", "again", "also", "answer", "before", "between", "could", "does",
    "from", "have", "into", "more", "than", "that", "their", "there", "these", "this",
    "what", "when", "where", "which", "with", "would", "tell", "explain", "summarize",
    "and", "has", "have", "had", "by", "supported",
}


def _payload(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) and result else result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(result: Any, key: str):
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _terms(text: str) -> set[str]:
    return {term.casefold() for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text) if term.casefold() not in _STOPWORDS}


def _context_terms(results: Iterable[Any] | None) -> set[str]:
    terms: set[str] = set()
    if results is None:
        return terms
    for result in results:
        for key in ("content", "text", "snippet", "title"):
            for value in _candidate_values(result, key):
                if (text := _string(value)):
                    terms.update(_terms(text))
    return terms


def check_query_answer_alignment(query: str, answer: str, results: Iterable[Any] | None = None) -> dict[str, Any]:
    """Return a compact alignment score for query, answer, and optional context."""
    query_terms = _terms(query)
    answer_terms = _terms(answer)
    context_terms = _context_terms(results)
    missing = sorted(query_terms - answer_terms)
    query_overlap = len(query_terms & answer_terms) / max(len(query_terms), 1)
    answer_focus_terms = answer_terms - query_terms
    unsupported = sorted(answer_focus_terms - context_terms)[:20] if context_terms else sorted(answer_focus_terms)[:20]
    support_ratio = 1.0 if not answer_focus_terms else 1 - (len(unsupported) / len(answer_focus_terms))
    score = round((query_overlap * 0.7) + (support_ratio * 0.3), 6)
    if score >= 0.75 and not missing:
        label = "aligned"
    elif score >= 0.45:
        label = "partially-aligned"
    else:
        label = "misaligned"
    return {
        "alignment_score": score,
        "label": label,
        "missing_query_terms": missing,
        "unsupported_answer_terms": unsupported,
        "summary": {
            "query_term_count": len(query_terms),
            "answer_term_count": len(answer_terms),
            "context_term_count": len(context_terms),
        },
    }
