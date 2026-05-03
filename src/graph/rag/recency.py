"""Freshness-aware reranking helpers for RAG result dictionaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
from math import pow
from typing import Any

_MISSING = object()
_SCORE_KEYS = (
    "score",
    "final_score",
    "hybrid_score",
    "semantic_score",
    "similarity",
    "fulltext_score",
)


def _validate_date_keys(date_keys: Sequence[str]) -> tuple[str, ...]:
    if isinstance(date_keys, str) or not isinstance(date_keys, Sequence):
        raise ValueError("date_keys must be a non-empty sequence of strings")
    keys = tuple(date_keys)
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("date_keys must be a non-empty sequence of strings")
    return keys


def _validate_half_life_days(half_life_days: int) -> int:
    if (
        not isinstance(half_life_days, int)
        or isinstance(half_life_days, bool)
        or half_life_days < 1
    ):
        raise ValueError("half_life_days must be a positive integer")
    return half_life_days


def _validate_score_weight(score_weight: float) -> float:
    if isinstance(score_weight, bool) or not isinstance(score_weight, int | float):
        raise ValueError("score_weight must be a number between 0 and 1")
    value = float(score_weight)
    if value < 0.0 or value > 1.0:
        raise ValueError("score_weight must be between 0 and 1")
    return value


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer or None")
    return limit


def _coerce_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if not isinstance(now, datetime):
        raise ValueError("now must be a datetime or None")
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _result_value(result: Any, key: str) -> Any:
    value = _field_value(result, key)
    if value is not _MISSING and value is not None:
        return value

    metadata = _field_value(result, "metadata")
    if isinstance(metadata, Mapping):
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING and metadata_value is not None:
            return metadata_value

    unit = _field_value(result, "unit")
    if unit is not _MISSING and unit is not None:
        unit_value = _field_value(unit, key)
        if unit_value is not _MISSING and unit_value is not None:
            return unit_value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            return unit_metadata.get(key, _MISSING)

    return value


def _parse_datetime(value: Any) -> datetime | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed_date = date.fromisoformat(text)
            except ValueError:
                return None
            parsed = datetime(
                parsed_date.year,
                parsed_date.month,
                parsed_date.day,
                tzinfo=timezone.utc,
            )
    else:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _result_datetime(result: Mapping[str, Any], date_keys: Sequence[str]) -> datetime | None:
    dates = [
        parsed
        for key in date_keys
        if (parsed := _parse_datetime(_result_value(result, key))) is not None
    ]
    if not dates:
        return None
    return max(dates)


def _numeric_value(value: Any) -> float | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _mapping_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, _MISSING)
    return _MISSING


def _relevance_score(result: Mapping[str, Any]) -> float:
    for key in _SCORE_KEYS:
        score = _numeric_value(_result_value(result, key))
        if score is not None:
            return score

    for container in (
        _result_value(result, "scores"),
        _mapping_value(_result_value(result, "explanation"), "scores"),
    ):
        if not isinstance(container, Mapping):
            continue
        for key in _SCORE_KEYS:
            score = _numeric_value(container.get(key, _MISSING))
            if score is not None:
                return score

    return 0.0


def _recency_score(timestamp: datetime | None, now: datetime, half_life_days: int) -> float:
    if timestamp is None:
        return 0.0
    age_seconds = max((now - timestamp).total_seconds(), 0.0)
    age_days = age_seconds / 86_400.0
    return pow(0.5, age_days / half_life_days)


def _string_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _identity(result: Mapping[str, Any], index: int) -> tuple[str, ...]:
    values = [
        _string_value(_result_value(result, key))
        for key in ("id", "unit_id", "source_id", "title", "url")
    ]
    return tuple(value for value in values if value) or (f"result-{index + 1}",)


def rerank_for_recency(
    results: Iterable[dict],
    *,
    date_keys: Sequence[str] = ("created_at", "updated_at", "published_at"),
    half_life_days: int = 90,
    score_weight: float = 0.75,
    now: datetime | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Blend search relevance with recency metadata and return reranked copies.

    Existing result dictionaries are not mutated. Missing or invalid dates get a
    recency score of ``0.0`` and sort after dated peers when relevance ties.
    """
    key_values = _validate_date_keys(date_keys)
    half_life = _validate_half_life_days(half_life_days)
    relevance_weight = _validate_score_weight(score_weight)
    limit_value = _validate_limit(limit)
    now_value = _coerce_now(now)

    scored = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError("results must contain dictionaries")

        timestamp = _result_datetime(result, key_values)
        relevance = _relevance_score(result)
        recency = _recency_score(timestamp, now_value, half_life)
        combined = (relevance_weight * relevance) + ((1.0 - relevance_weight) * recency)
        payload = {
            **result,
            "recency_score": round(recency, 6),
            "combined_score": round(combined, 6),
        }
        scored.append(
            (
                -combined,
                timestamp is None,
                -recency,
                _identity(result, index),
                index,
                payload,
            )
        )

    reranked = [payload for *_sort_key, payload in sorted(scored)]
    if limit_value is not None:
        return reranked[:limit_value]
    return reranked
