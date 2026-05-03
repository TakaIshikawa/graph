"""GEXF export helpers for knowledge graphs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel

from graph.types.models import KnowledgeEdge, KnowledgeUnit


def export_graph_gexf(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path,
) -> dict[str, Any]:
    """Write units and valid edges as a deterministic GEXF directed graph."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_graph = nx.DiGraph()
    exported_units = sorted(units, key=lambda unit: _text(unit.id))
    unit_ids = {_text(unit.id) for unit in exported_units}

    for unit in exported_units:
        export_graph.add_node(_text(unit.id), **_node_attributes(unit))

    skipped_edges: list[dict[str, str]] = []
    exported_edges = sorted(
        edges,
        key=lambda edge: (
            _text(edge.from_unit_id),
            _text(edge.to_unit_id),
            _field_value(edge.relation),
            _text(edge.id),
        ),
    )
    for edge in exported_edges:
        missing = [
            unit_id
            for unit_id in (_text(edge.from_unit_id), _text(edge.to_unit_id))
            if unit_id not in unit_ids
        ]
        if missing:
            skipped_edges.append(
                {
                    "id": _text(edge.id),
                    "from_unit_id": _text(edge.from_unit_id),
                    "to_unit_id": _text(edge.to_unit_id),
                    "reason": f"missing_units:{','.join(sorted(set(missing)))}",
                }
            )
            continue
        export_graph.add_edge(
            _text(edge.from_unit_id),
            _text(edge.to_unit_id),
            **_edge_attributes(edge),
        )

    nx.write_gexf(export_graph, output_path)
    return {
        "path": str(output_path),
        "units_exported": export_graph.number_of_nodes(),
        "edges_exported": export_graph.number_of_edges(),
        "skipped_edges": skipped_edges,
    }


def _node_attributes(unit: KnowledgeUnit) -> dict[str, Any]:
    return {
        "label": _text(unit.title),
        "title": _text(unit.title),
        "source_project": _field_value(unit.source_project),
        "source_id": _text(unit.source_id),
        "source_entity_type": _text(unit.source_entity_type),
        "content_type": _field_value(unit.content_type),
        "tags": ",".join(_text(tag) for tag in unit.tags),
        "metadata": _json_text(unit.metadata),
        "confidence": float(unit.confidence or 0.0),
        "utility_score": float(unit.utility_score or 0.0),
        "created_at": _json_value(unit.created_at),
        "updated_at": _json_value(unit.updated_at),
    }


def _edge_attributes(edge: KnowledgeEdge) -> dict[str, Any]:
    return {
        "id": _text(edge.id),
        "relation": _field_value(edge.relation),
        "weight": float(edge.weight or 0.0),
        "source": _field_value(edge.source),
        "metadata": _json_text(edge.metadata),
        "created_at": _json_value(edge.created_at),
    }


def _json_text(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True)


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
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=_item_key)
        }
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return str(value)


def _item_key(item: tuple[Any, Any]) -> str:
    return str(item[0])


def _field_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _text(value: object) -> str:
    return str(value or "")
