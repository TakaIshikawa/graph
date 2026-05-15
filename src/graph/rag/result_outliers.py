"""Detect weakly related outliers in retrieved RAG/search results."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_TEXT_KEYS = ("title", "content", "text", "snippet", "tags")
_ID_KEYS = ("id", "unit_id", "source_id")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(result: Any, key: str) -> Iterable[Any]:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value
    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        value = _field_value(unit, key)
        if value is not _MISSING:
            yield value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            value = unit_metadata.get(key, _MISSING)
            if value is not _MISSING:
                yield value


def _strings(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Mapping):
        return [str(item) for pair in value.items() for item in pair if item is not None]
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for text in _strings(value):
        tokens.update(
            token
            for token in re.findall(r"[a-z0-9][a-z0-9-]*", text.casefold())
            if len(token) > 1 and token not in _STOP_WORDS
        )
    return tokens


def _result_tokens(result: Any) -> set[str]:
    tokens: set[str] = set()
    for key in _TEXT_KEYS:
        for value in _candidate_values(result, key):
            tokens.update(_tokens(value))
    return tokens


def _text(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).split())
    return text or None


def _result_id(result: Any, index: int) -> str:
    for key in _ID_KEYS:
        for value in _candidate_values(result, key):
            text = _text(value)
            if text is not None:
                return text
    return f"result-{index + 1}"


def detect_result_outliers(
    results: Iterable[Any],
    *,
    min_overlap: float = 0.15,
) -> dict[str, Any]:
    """Return retrieval results whose tokens weakly overlap with the set baseline."""
    if not isinstance(min_overlap, int | float) or isinstance(min_overlap, bool) or min_overlap < 0:
        raise ValueError("min_overlap must be a non-negative number")

    rows = [
        {"index": index, "result": result, "result_id": _result_id(result, index), "tokens": _result_tokens(result)}
        for index, result in enumerate(results)
    ]
    token_counts = Counter(token for row in rows for token in row["tokens"])
    baseline_terms = sorted(token for token, count in token_counts.items() if count >= 2)
    baseline = set(baseline_terms)

    outliers: list[dict[str, Any]] = []
    for row in rows:
        tokens = row["tokens"]
        shared = tokens & baseline
        score = len(shared) / len(tokens) if tokens else 0.0
        if score >= min_overlap:
            continue
        distinctive = tokens - baseline
        reason = "no comparable tokens" if not tokens else "low token overlap with retrieved result set"
        outliers.append(
            {
                "result_id": row["result_id"],
                "overlap_score": round(score, 3),
                "shared_terms": sorted(shared),
                "distinctive_terms": sorted(distinctive)[:10],
                "reason": reason,
            }
        )

    return {
        "result_count": len(rows),
        "baseline_terms": baseline_terms,
        "token_frequencies": dict(sorted(token_counts.items())),
        "outliers": outliers,
    }
