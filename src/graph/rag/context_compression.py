"""Plan deterministic context compression for RAG/search results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_ID_KEYS = ("id", "unit_id", "source_id")
_TEXT_KEYS = ("snippet", "content", "text", "summary", "title")
_SCORE_KEYS = ("confidence", "score", "final_score", "hybrid_score", "semantic_score")
_WORD_RE = re.compile(r"\S+")


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


def _result_value(result: Any, key: str) -> Any:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value

    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING and value is not None:
            return value

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        value = _field_value(unit, key)
        if value is not _MISSING and value is not None:
            return value
        metadata = _field_value(unit, "metadata")
        if isinstance(metadata, Mapping):
            return metadata.get(key, _MISSING)

    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping):
        text = " ".join(str(item).strip() for item in value if str(item).strip())
    else:
        text = str(value)
    text = " ".join(text.strip().split())
    return text or None


def _result_id(result: Any, index: int) -> str:
    for key in _ID_KEYS:
        value = _string(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _text(result: Any) -> str:
    parts = []
    for key in _TEXT_KEYS:
        value = _string(_result_value(result, key))
        if value is not None and value not in parts:
            parts.append(value)
    return " ".join(parts)


def _token_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _numeric(value: Any) -> float | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _priority(result: Any) -> float:
    for key in _SCORE_KEYS:
        value = _numeric(_result_value(result, key))
        if value is not None:
            return min(max(value if value <= 1 else value / 100, 0.0), 1.0)
    return 0.25


def _validate_token_budget(token_budget: int) -> int:
    if not isinstance(token_budget, int) or isinstance(token_budget, bool) or token_budget < 0:
        raise ValueError("token_budget must be a non-negative integer")
    return token_budget


def _candidate(result: Any, index: int) -> dict[str, Any]:
    text = _text(result)
    estimated_tokens = _token_count(text)
    priority = _priority(result)
    return {
        "result_id": _result_id(result, index),
        "index": index,
        "priority": priority,
        "estimated_tokens": estimated_tokens,
        "text": text,
    }


def plan_context_compression(results: Iterable[Any], *, token_budget: int) -> dict[str, Any]:
    """Allocate a token budget across higher-priority result snippets/content."""
    budget = _validate_token_budget(token_budget)
    candidates = [_candidate(result, index) for index, result in enumerate(results)]
    ranked = sorted(candidates, key=lambda item: (-item["priority"], item["index"]))

    allocations = []
    dropped = []
    remaining = budget

    for item in ranked:
        if remaining <= 0:
            dropped.append({"result_id": item["result_id"], "reason": "token budget exhausted"})
            continue
        if item["estimated_tokens"] <= 0:
            dropped.append({"result_id": item["result_id"], "reason": "no text to include"})
            continue

        minimum = 1 if budget < 8 else min(8, item["estimated_tokens"])
        if remaining < minimum:
            dropped.append({"result_id": item["result_id"], "reason": "insufficient remaining budget"})
            continue

        desired = max(minimum, round(item["estimated_tokens"] * (0.35 + item["priority"] * 0.5)))
        allocated = min(item["estimated_tokens"], desired, remaining)
        remaining -= allocated
        allocations.append(
            {
                "result_id": item["result_id"],
                "allocated_tokens": allocated,
                "estimated_tokens": item["estimated_tokens"],
                "priority": round(item["priority"], 3),
                "action": "include" if allocated >= item["estimated_tokens"] else "trim",
            }
        )

    allocations.sort(key=lambda item: (-item["priority"], item["result_id"]))
    return {
        "token_budget": budget,
        "allocated_tokens": sum(item["allocated_tokens"] for item in allocations),
        "remaining_tokens": budget - sum(item["allocated_tokens"] for item in allocations),
        "allocations": allocations,
        "dropped_result_ids": [item["result_id"] for item in dropped],
        "dropped": dropped,
    }
