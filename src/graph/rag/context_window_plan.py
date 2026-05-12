"""Plan a deterministic RAG context window within a token budget."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import TOKEN_RE

_MISSING = object()
_TEXT_KEYS = ("title", "content", "summary")


def _validate_non_negative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


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


def _tuple_score(result: Any) -> Any:
    if isinstance(result, tuple) and len(result) > 1:
        return result[1]
    return _MISSING


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
    if key == "score":
        return _tuple_score(result)
    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _float(value: Any) -> float:
    if value is _MISSING or value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string(_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _source(result: Any) -> str:
    return _string(_value(result, "source_project")) or "unknown"


def _estimated_tokens(result: Any) -> int:
    """Estimate tokens as the count of word-like tokens, with a one-token floor."""
    text = " ".join(_string(_value(result, key)) or "" for key in _TEXT_KEYS)
    return max(1, len(TOKEN_RE.findall(text)))


def _summary(candidate: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "id": candidate["id"],
        "source_project": candidate["source_project"],
        "estimated_tokens": candidate["estimated_tokens"],
        "score": candidate["score"],
        "reason": reason,
    }


def plan_context_window(
    results: Iterable[Any],
    *,
    token_budget: int,
    reserve_tokens: int = 0,
    min_per_source: int = 0,
) -> dict[str, Any]:
    """Produce an inclusion plan that never exceeds the available token budget."""
    budget = _validate_non_negative_int(token_budget, "token_budget")
    reserve = _validate_non_negative_int(reserve_tokens, "reserve_tokens")
    min_source = _validate_non_negative_int(min_per_source, "min_per_source")
    available = max(0, budget - reserve)
    candidates = [
        {
            "id": _id(result, index),
            "source_project": _source(result),
            "estimated_tokens": _estimated_tokens(result),
            "score": _float(_value(result, "score")),
            "rank": index,
        }
        for index, result in enumerate(results)
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    used = 0

    if min_source:
        for source in sorted({item["source_project"] for item in candidates}):
            source_items = sorted(
                [item for item in candidates if item["source_project"] == source],
                key=lambda item: (-item["score"], item["rank"], item["id"]),
            )
            for item in source_items[:min_source]:
                if item["id"] not in selected_ids and used + item["estimated_tokens"] <= available:
                    selected.append(item)
                    selected_ids.add(item["id"])
                    source_counts[source] += 1
                    used += item["estimated_tokens"]

    remaining = [item for item in candidates if item["id"] not in selected_ids]
    remaining.sort(key=lambda item: (-item["score"], source_counts[item["source_project"]], item["rank"], item["id"]))
    for item in remaining:
        if used + item["estimated_tokens"] > available:
            continue
        selected.append(item)
        selected_ids.add(item["id"])
        source_counts[item["source_project"]] += 1
        used += item["estimated_tokens"]

    included = [_summary(item, "included") for item in sorted(selected, key=lambda item: item["rank"])]
    excluded = [
        _summary(item, "over_budget" if used + item["estimated_tokens"] > available else "not_selected")
        for item in candidates
        if item["id"] not in selected_ids
    ]
    return {
        "included": included,
        "excluded": excluded,
        "used_tokens": used,
        "available_tokens": available,
        "overflow_count": len(excluded),
    }
