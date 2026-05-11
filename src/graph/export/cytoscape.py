"""Cytoscape.js JSON export for graph visualization."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeEdge, KnowledgeUnit


def export_graph_cytoscape(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    include_content: bool = False,
) -> str | dict[str, Any]:
    """Return or write a deterministic Cytoscape.js elements JSON graph."""
    unit_list = [units] if isinstance(units, KnowledgeUnit) else list(units)
    exported_units = sorted(unit_list, key=_unit_key)
    unit_ids = {_unit_id(unit) for unit in exported_units}

    exported_edges: list[KnowledgeEdge] = []
    skipped_edge_count = 0
    for edge in sorted(edges, key=_edge_key):
        if _text(edge.from_unit_id) in unit_ids and _text(edge.to_unit_id) in unit_ids:
            exported_edges.append(edge)
        else:
            skipped_edge_count += 1

    graph = {
        "elements": {
            "nodes": [_node(unit, include_content=include_content) for unit in exported_units],
            "edges": [_edge(edge) for edge in exported_edges],
        }
    }
    text = json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "node_count": len(exported_units),
        "edge_count": len(exported_edges),
        "skipped_edge_count": skipped_edge_count,
        "bytes_written": len(text.encode("utf-8")),
    }


def _node(unit: KnowledgeUnit, *, include_content: bool) -> dict[str, Any]:
    data = {
        "id": _unit_id(unit),
        "label": _text(unit.title),
        "source_project": _field_value(unit.source_project),
        "source_id": _text(unit.source_id),
        "source_entity_type": _text(unit.source_entity_type),
        "content_type": _field_value(unit.content_type),
        "tags": _json_value(unit.tags),
        "metadata": _json_value(unit.metadata),
    }
    if include_content:
        data["content"] = _text(unit.content)
    return {"data": data}


def _edge(edge: KnowledgeEdge) -> dict[str, Any]:
    relation = _field_value(edge.relation)
    return {
        "data": {
            "id": _edge_id(edge),
            "source": _text(edge.from_unit_id),
            "target": _text(edge.to_unit_id),
            "label": relation,
            "relation": relation,
            "weight": edge.weight,
            "edge_source": _field_value(edge.source),
            "metadata": _json_value(edge.metadata),
        }
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


def _unit_id(unit: KnowledgeUnit) -> str:
    return _text(unit.id or unit.source_id)


def _edge_id(edge: KnowledgeEdge) -> str:
    edge_id = _text(edge.id)
    if edge_id:
        return edge_id
    return "|".join((_text(edge.from_unit_id), _text(edge.to_unit_id), _field_value(edge.relation)))


def _unit_key(unit: KnowledgeUnit) -> tuple[str, str]:
    return (_unit_id(unit), _text(unit.source_id))


def _edge_key(edge: KnowledgeEdge) -> tuple[str, str, str, str]:
    return (
        _text(edge.from_unit_id),
        _text(edge.to_unit_id),
        _field_value(edge.relation),
        _edge_id(edge),
    )


def _field_value(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return _text(value)


def _text(value: object) -> str:
    return str(value or "")
