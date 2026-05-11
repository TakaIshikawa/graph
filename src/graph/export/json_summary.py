"""Compact JSON summary export for graph data."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_TIMESTAMP_KEYS = ("timestamp", "created_at", "updated_at", "ingested_at", "published_at")


def export_graph_json_summary(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
) -> str:
    """Return graph counts and top-level metadata as deterministic JSON."""
    unit_list = list(units)
    edge_list = list(edges)

    timestamps = [_timestamp_text(value) for unit in unit_list for value in _unit_timestamps(unit)]
    timestamps.extend(_timestamp_text(value) for edge in edge_list for value in _edge_timestamps(edge))
    timestamp_values = sorted(value for value in timestamps if value)

    summary: dict[str, Any] = {
        "total_nodes": len(unit_list),
        "total_edges": len(edge_list),
        "node_types": dict(sorted(Counter(_node_type(unit) for unit in unit_list).items())),
        "edge_types": dict(sorted(Counter(_edge_type(edge) for edge in edge_list).items())),
        "tag_counts": dict(sorted(_tag_counts(unit_list).items())),
    }
    if timestamp_values:
        summary["min_timestamp"] = timestamp_values[0]
        summary["max_timestamp"] = timestamp_values[-1]

    return json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _node_type(unit: KnowledgeUnit) -> str:
    return _text(unit.source_entity_type or unit.content_type)


def _edge_type(edge: KnowledgeEdge) -> str:
    return _text(getattr(edge.relation, "value", edge.relation))


def _tag_counts(units: Iterable[KnowledgeUnit]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for unit in units:
        for tag in unit.tags:
            text = _text(tag).strip()
            if text:
                counts[text] += 1
    return counts


def _unit_timestamps(unit: KnowledgeUnit) -> list[Any]:
    values: list[Any] = [unit.created_at, unit.ingested_at, unit.updated_at]
    values.extend(_metadata_timestamps(unit.metadata))
    return values


def _edge_timestamps(edge: KnowledgeEdge) -> list[Any]:
    return [edge.created_at, *_metadata_timestamps(edge.metadata)]


def _metadata_timestamps(metadata: Mapping[str, Any]) -> list[Any]:
    values: list[Any] = []
    for key, value in metadata.items():
        if key in _TIMESTAMP_KEYS or key.endswith("_at") or key.endswith("_date"):
            values.append(value)
        if isinstance(value, Mapping):
            values.extend(_metadata_timestamps(value))
    return values


def _timestamp_text(value: Any) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, str):
        return value.strip()
    return ""


def _text(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value or "")
