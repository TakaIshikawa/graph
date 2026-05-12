"""Score incremental novelty for ordered RAG retrieval results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_DEFAULT_TEXT_KEYS = ("title", "content", "text", "summary")


def _validate_threshold(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("similarity_threshold must be between 0 and 1")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("similarity_threshold must be between 0 and 1") from exc
    if threshold < 0 or threshold > 1:
        raise ValueError("similarity_threshold must be between 0 and 1")
    return threshold


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _value(result: Any, key: str) -> Any:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING and metadata_value is not None:
            return metadata_value
    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        unit_value = _field_value(unit, key)
        if unit_value is not _MISSING and unit_value is not None:
            return unit_value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            return unit_metadata.get(key, _MISSING)
    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _keys(text_keys: Iterable[str] | None) -> tuple[str, ...]:
    if text_keys is None:
        return _DEFAULT_TEXT_KEYS
    keys = tuple(text_keys)
    for key in keys:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("text_keys must contain non-empty strings")
    return tuple(key.strip() for key in keys)


def _id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string(_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _source(result: Any) -> str:
    return _string(_value(result, "source_project")) or "unknown"


def _tokens(result: Any, keys: tuple[str, ...]) -> set[str]:
    terms: set[str] = set()
    for key in keys:
        text = _string(_value(result, key))
        if text:
            terms.update(
                token
                for token in TOKEN_RE.findall(text.casefold())
                if len(token) > 1 and token not in COMMON_STOPWORDS
            )
    return terms


def score_retrieval_novelty(
    results: Iterable[Any],
    *,
    text_keys: Iterable[str] | None = None,
    similarity_threshold: float = 0.65,
) -> list[dict[str, Any]]:
    """Score each result's new information against earlier ranked results."""
    threshold = _validate_threshold(similarity_threshold)
    keys = _keys(text_keys)
    previous: list[dict[str, Any]] = []
    rows = []

    for index, result in enumerate(results):
        terms = _tokens(result, keys)
        best: dict[str, Any] | None = None
        for prior in previous:
            denominator = max(min(len(terms), len(prior["terms"])), 1)
            shared = sorted(terms & prior["terms"])
            similarity = len(shared) / denominator
            candidate = {"id": prior["id"], "similarity": similarity, "shared": shared}
            if best is None or (candidate["similarity"], -previous.index(prior)) > (
                best["similarity"],
                -best["rank"],
            ):
                candidate["rank"] = prior["rank"]
                best = candidate
        duplicate_of = best["id"] if best is not None and best["similarity"] >= threshold else None
        shared_terms = [] if best is None else best["shared"]
        rows.append(
            {
                "id": _id(result, index),
                "source_project": _source(result),
                "novelty_score": round(1 - (best["similarity"] if best else 0), 6),
                "duplicate_of": duplicate_of,
                "shared_terms": shared_terms,
                "token_count": len(terms),
            }
        )
        previous.append({"id": _id(result, index), "terms": terms, "rank": index})
    return rows
