"""GraphSON-compatible JSON export helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeEdge, KnowledgeUnit


def export_graphson(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path,
    *,
    directed: bool = True,
) -> dict:
    """Write units and edges as deterministic GraphSON-style JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported_units = sorted(units, key=lambda unit: _text(unit.id))
    exported_edges = sorted(
        edges,
        key=lambda edge: (
            _text(edge.from_unit_id),
            _text(edge.to_unit_id),
            _field_value(edge.relation),
            _text(edge.id),
        ),
    )

    graph = {
        "directed": directed,
        "nodes": [_node(unit) for unit in exported_units],
        "edges": [_edge(edge) for edge in exported_edges],
    }
    output_path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "path": str(output_path),
        "node_count": len(exported_units),
        "edge_count": len(exported_edges),
        "directed": directed,
    }


def _node(unit: KnowledgeUnit) -> dict[str, Any]:
    return {
        "id": _text(unit.id),
        "label": "knowledge_unit",
        "properties": {
            "source_project": _field_value(unit.source_project),
            "source_id": _text(unit.source_id),
            "source_entity_type": _text(unit.source_entity_type),
            "title": _text(unit.title),
            "content": _text(unit.content),
            "content_type": _field_value(unit.content_type),
            "metadata": _json_value(unit.metadata),
            "tags": _json_value(unit.tags),
            "confidence": unit.confidence,
            "utility_score": unit.utility_score,
            "created_at": _json_value(unit.created_at),
            "ingested_at": _json_value(unit.ingested_at),
            "updated_at": _json_value(unit.updated_at),
        },
    }


def _edge(edge: KnowledgeEdge) -> dict[str, Any]:
    return {
        "id": _text(edge.id),
        "source": _text(edge.from_unit_id),
        "target": _text(edge.to_unit_id),
        "label": _field_value(edge.relation),
        "properties": {
            "relation": _field_value(edge.relation),
            "weight": edge.weight,
            "source": _field_value(edge.source),
            "metadata": _json_value(edge.metadata),
            "created_at": _json_value(edge.created_at),
        },
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=_item_key)}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return str(value)


def _item_key(item: tuple[Any, Any]) -> str:
    return str(item[0])


def _field_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _text(value: object) -> str:
    return str(value or "")
