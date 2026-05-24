"""Summarize unit embedding metadata coverage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def unit_embedding_coverage_summary(units: Iterable[Any]) -> list[dict[str, Any]]:
    rows: dict[tuple[str | None, str | None, int | None, str], dict[str, Any]] = {}
    for unit in units:
        metadata = _metadata(unit)
        embedding = metadata.get("embedding")
        blob_embedding = _get(unit, "embedding")
        updated_at = _string(_get(unit, "updated_at"))
        if isinstance(embedding, Mapping):
            provider = _string(embedding.get("provider"))
            model = _string(embedding.get("model"))
            dimension = _dimension(embedding.get("dimension") or embedding.get("dimensions") or embedding.get("vector"))
            embedded_at = _string(
                embedding.get("updated_at") or embedding.get("embedded_at") or embedding.get("created_at")
            )
            has_embedding = True
        elif blob_embedding is not None:
            provider = model = None
            dimension = None
            embedded_at = None
            has_embedding = True
        else:
            provider = model = None
            dimension = None
            embedded_at = None
            has_embedding = False
        status = "missing"
        if has_embedding:
            status = "malformed" if isinstance(embedding, Mapping) and dimension is None else "current"
            stale_marker = metadata.get("embedding_stale")
            if stale_marker is True or (embedded_at and updated_at and embedded_at < updated_at):
                status = "stale"
        key = (provider, model, dimension, status)
        rows.setdefault(
            key,
            {
                "provider": provider,
                "model": model,
                "dimension": dimension,
                "status": status,
                "count": 0,
            },
        )["count"] += 1
    return sorted(
        rows.values(),
        key=lambda row: (
            row["status"],
            row["provider"] or "",
            row["model"] or "",
            -1 if row["dimension"] is None else row["dimension"],
        ),
    )


def _dimension(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    try:
        dimension = int(value)
    except (TypeError, ValueError):
        return None
    return dimension if dimension > 0 else None


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
