"""Summarize unit embedding provider, model, and dimension mix."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_PROVIDER_KEYS = ("embedding_provider", "provider")
_MODEL_KEYS = ("embedding_model", "model")
_DIMENSION_KEYS = ("embedding_dimensions", "embedding_dimension", "dimensions", "dimension", "vector")


def unit_embedding_model_mix_summary(units: Iterable[Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    total = 0
    for unit in units:
        total += 1
        embedding = _embedding(unit)
        provider = _string(_first(unit, embedding, _PROVIDER_KEYS)) or "unknown"
        model = _string(_first(unit, embedding, _MODEL_KEYS)) or "unknown"
        dimensions = _dimension(_first(unit, embedding, _DIMENSION_KEYS))
        key = (provider, model, dimensions)
        group = groups.setdefault(
            key,
            {
                "provider": provider,
                "model": model,
                "dimensions": dimensions,
                "count": 0,
                "sample_unit_ids": [],
            },
        )
        group["count"] += 1
        unit_id = _unit_id(unit)
        if unit_id and len(group["sample_unit_ids"]) < 3:
            group["sample_unit_ids"].append(unit_id)

    rows = []
    for group in groups.values():
        rows.append(
            {
                "provider": group["provider"],
                "model": group["model"],
                "dimensions": group["dimensions"],
                "count": group["count"],
                "share": round(group["count"] / total, 4) if total else 0,
                "sample_unit_ids": group["sample_unit_ids"],
            }
        )
    return sorted(rows, key=lambda row: (-row["count"], row["provider"], row["model"], row["dimensions"] or -1))


def _embedding(item: Any) -> Mapping[str, Any]:
    metadata = _metadata(item)
    value = metadata.get("embedding")
    if isinstance(value, Mapping):
        return value
    return metadata


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _first(item: Any, metadata: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _get(item, key)
        if value not in (None, ""):
            return value
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _unit_id(item: Any) -> str | None:
    return _string(_get(item, "id") or _get(item, "unit_id") or _metadata(item).get("id"))


def _dimension(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    try:
        dimension = int(value)
    except (TypeError, ValueError):
        return None
    return dimension if dimension > 0 else None


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
