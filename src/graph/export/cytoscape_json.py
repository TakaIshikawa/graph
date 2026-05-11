"""Cytoscape.js elements JSON export helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, overload

from pydantic import BaseModel

from graph.types.models import KnowledgeEdge, KnowledgeUnit


@overload
def export_graph_cytoscape_json(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: None = None,
    *,
    include_content: bool = False,
) -> dict[str, Any]: ...


@overload
def export_graph_cytoscape_json(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path,
    *,
    include_content: bool = False,
) -> dict[str, Any]: ...


def export_graph_cytoscape_json(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    include_content: bool = False,
) -> dict[str, Any]:
    """Return or write Cytoscape.js-compatible elements with data payloads."""
    unit_list = sorted(list(units), key=_unit_key)
    edge_list = sorted(list(edges), key=_edge_key)
    elements = {
        "elements": {
            "nodes": [_node(unit, include_content=include_content) for unit in unit_list],
            "edges": [_edge(edge) for edge in edge_list],
        }
    }

    if path is None:
        return elements

    text = json.dumps(elements, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "node_count": len(unit_list),
        "edge_count": len(edge_list),
        "bytes_written": output_path.stat().st_size,
    }


def _node(unit: KnowledgeUnit, *, include_content: bool) -> dict[str, Any]:
    data = {
        "id": _unit_id(unit),
        "label": _text(unit.title),
        "type": _node_type(unit),
        "source_project": _json_value(unit.source_project),
        "source_id": _text(unit.source_id),
        "source_entity_type": _text(unit.source_entity_type),
        "content_type": _json_value(unit.content_type),
        "tags": _json_value(unit.tags),
        "metadata": _json_value(unit.metadata),
    }
    if include_content:
        data["content"] = _text(unit.content)
    return {"data": data}


def _edge(edge: KnowledgeEdge) -> dict[str, Any]:
    relation = _json_value(edge.relation)
    return {
        "data": {
            "id": _edge_id(edge),
            "source": _text(edge.from_unit_id),
            "target": _text(edge.to_unit_id),
            "label": relation,
            "type": relation,
            "relation": relation,
            "weight": edge.weight,
            "edge_source": _json_value(edge.source),
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
    if isinstance(value, list | tuple | set):
        return [_json_value(item) for item in value]
    return str(value)


def _unit_id(unit: KnowledgeUnit) -> str:
    return _text(unit.id or unit.source_id)


def _edge_id(edge: KnowledgeEdge) -> str:
    edge_id = _text(edge.id)
    if edge_id:
        return edge_id
    return "|".join((_text(edge.from_unit_id), _text(edge.to_unit_id), _text(_json_value(edge.relation))))


def _node_type(unit: KnowledgeUnit) -> str:
    return _text(unit.source_entity_type or _json_value(unit.content_type))


def _unit_key(unit: KnowledgeUnit) -> tuple[str, str]:
    return (_unit_id(unit), _text(unit.source_id))


def _edge_key(edge: KnowledgeEdge) -> tuple[str, str, str, str]:
    return (
        _text(edge.from_unit_id),
        _text(edge.to_unit_id),
        _text(_json_value(edge.relation)),
        _edge_id(edge),
    )


def _item_key(item: tuple[Any, Any]) -> str:
    return str(item[0])


def _text(value: object) -> str:
    return str(value or "")
