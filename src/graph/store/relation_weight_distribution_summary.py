"""Summarize relation weight distributions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_RELATION_KEYS = ("relation", "relation_type", "type", "predicate")
_SOURCE_KEYS = ("source", "edge_source", "source_project")
_WEIGHT_KEYS = ("weight", "score", "confidence")


def summarize_relation_weight_distribution(edges: Iterable[Any]) -> dict[str, Any]:
    """Group edges by relation and source and summarize valid numeric weights."""

    groups: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    total_edges = 0
    for edge in edges:
        total_edges += 1
        metadata = _metadata(edge)
        relation = _string(_first(edge, metadata, _RELATION_KEYS))
        source = _string(_first(edge, metadata, _SOURCE_KEYS))
        key = (relation, source)
        group = groups.setdefault(
            key,
            {"relation": relation, "source": source, "edge_count": 0, "missing_weight_count": 0, "weights": []},
        )
        group["edge_count"] += 1
        weight = _number(_first(edge, metadata, _WEIGHT_KEYS))
        if weight is None:
            group["missing_weight_count"] += 1
        else:
            group["weights"].append(weight)

    rows = []
    for key in sorted(groups, key=lambda item: ((item[0] or ""), (item[1] or ""))):
        group = groups[key]
        weights = group["weights"]
        rows.append(
            {
                "relation": group["relation"],
                "source": group["source"],
                "edge_count": group["edge_count"],
                "missing_weight_count": group["missing_weight_count"],
                "min_weight": min(weights) if weights else None,
                "max_weight": max(weights) if weights else None,
                "average_weight": round(sum(weights) / len(weights), 4) if weights else None,
                "zero_weight_count": sum(1 for weight in weights if weight == 0),
            }
        )
    return {"rows": rows, "row_count": len(rows), "edge_count": total_edges}


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


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
