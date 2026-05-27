"""Summarize reciprocal relation coverage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_RELATION_KEYS = ("relation", "relation_type", "type", "predicate")
_SOURCE_KEYS = ("source", "edge_source", "source_project")
_FROM_KEYS = ("from_unit_id", "source_unit_id", "from_id")
_TO_KEYS = ("to_unit_id", "target_unit_id", "to_id")


def summarize_relation_reciprocity(edges: Iterable[Any]) -> dict[str, Any]:
    """Group edges by relation/source and count reciprocal directed relationships."""

    groups: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    total_edges = 0
    for edge in edges:
        total_edges += 1
        metadata = _metadata(edge)
        relation = _string(_first(edge, metadata, _RELATION_KEYS))
        source = _string(_first(edge, metadata, _SOURCE_KEYS))
        group = groups.setdefault(
            (relation, source),
            {"relation": relation, "source": source, "edge_count": 0, "pairs": set(), "self_loop_count": 0},
        )
        group["edge_count"] += 1
        from_id = _string(_first(edge, metadata, _FROM_KEYS))
        to_id = _string(_first(edge, metadata, _TO_KEYS))
        if from_id is None or to_id is None:
            continue
        if from_id == to_id:
            group["self_loop_count"] += 1
        else:
            group["pairs"].add((from_id, to_id))

    rows = []
    for key in sorted(groups, key=lambda item: ((item[0] or ""), (item[1] or ""))):
        group = groups[key]
        pairs = group["pairs"]
        reciprocal_edges = sum(1 for from_id, to_id in pairs if (to_id, from_id) in pairs)
        one_way_edges = len(pairs) - reciprocal_edges
        valid_edges = reciprocal_edges + one_way_edges
        rows.append(
            {
                "relation": group["relation"],
                "source": group["source"],
                "edge_count": group["edge_count"],
                "reciprocal_edge_count": reciprocal_edges,
                "one_way_edge_count": one_way_edges,
                "self_loop_count": group["self_loop_count"],
                "reciprocal_ratio": round(reciprocal_edges / valid_edges, 4) if valid_edges else 0.0,
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


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
