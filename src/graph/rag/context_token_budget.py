"""Allocate context token budgets across retrieved RAG results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_ID_KEYS = ("id", "result_id", "unit_id", "source_id")
_SCORE_KEYS = ("score", "relevance_score", "rank_score", "density_score")
_SOURCE_KEYS = ("source", "source_name", "source_project", "domain", "publisher")
_DATE_KEYS = ("published_at", "publication_date", "updated_at", "created_at", "date", "timestamp")


def allocate_context_token_budget(
    results: Iterable[Any],
    total_budget: int,
    *,
    min_tokens_per_result: int = 0,
) -> list[dict[str, Any]]:
    """Assign deterministic token budgets without exceeding total_budget."""
    if not isinstance(total_budget, int) or isinstance(total_budget, bool) or total_budget < 0:
        raise ValueError("total_budget must be a non-negative integer")
    if not isinstance(min_tokens_per_result, int) or isinstance(min_tokens_per_result, bool) or min_tokens_per_result < 0:
        raise ValueError("min_tokens_per_result must be a non-negative integer")

    result_list = list(results)
    if not result_list:
        return []
    base = min(min_tokens_per_result, total_budget // len(result_list))
    allocations = [base for _ in result_list]
    remaining = total_budget - sum(allocations)
    weights = [_weight(result) for result in result_list]
    total_weight = sum(weights) or float(len(result_list))

    fractional = []
    for index, weight in enumerate(weights):
        exact = remaining * (weight / total_weight)
        extra = int(exact)
        allocations[index] += extra
        fractional.append((exact - extra, weight, index))
    leftover = total_budget - sum(allocations)
    for _, _, index in sorted(fractional, key=lambda item: (-item[0], -item[1], item[2]))[:leftover]:
        allocations[index] += 1

    return [
        {
            "result_id": _result_id(result, index),
            "allocated_tokens": allocation,
            "weight": round(weights[index], 3),
        }
        for index, (result, allocation) in enumerate(zip(result_list, allocations, strict=True))
    ]


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


def _value(result: Any, key: str) -> Any:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(key, _MISSING)
    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _result_id(result: Any, index: int) -> str:
    for key in _ID_KEYS:
        text = _string(_value(result, key))
        if text is not None:
            return text
    return f"result-{index + 1}"


def _score(result: Any) -> float:
    for key in _SCORE_KEYS:
        value = _value(result, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return max(float(value), 0.0)
        text = _string(value)
        if text is not None:
            try:
                return max(float(text), 0.0)
            except ValueError:
                continue
    return 0.0


def _weight(result: Any) -> float:
    score_weight = 1.0 + _score(result)
    source_bonus = 0.25 if any(_string(_value(result, key)) for key in _SOURCE_KEYS) else 0.0
    recency_bonus = 0.25 if any(_string(_value(result, key)) for key in _DATE_KEYS) else 0.0
    return score_weight + source_bonus + recency_bonus
