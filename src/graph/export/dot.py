"""Graphviz DOT export helpers for knowledge graphs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeEdge, KnowledgeUnit


def export_graph_dot(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    *,
    graph_name: str = "KnowledgeGraph",
    include_metadata: bool = False,
) -> str:
    """Return a deterministic Graphviz DOT representation of units and edges."""
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

    lines = [f"digraph {_dot_string(graph_name)} {{"]
    for unit in exported_units:
        lines.append(
            f"  {_dot_string(unit.id)} [{_attributes(_node_attributes(unit, include_metadata=include_metadata))}];"
        )
    for edge in exported_edges:
        lines.append(
            "  "
            f"{_dot_string(edge.from_unit_id)} -> {_dot_string(edge.to_unit_id)} "
            f"[{_attributes(_edge_attributes(edge, include_metadata=include_metadata))}];"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _node_attributes(unit: KnowledgeUnit, *, include_metadata: bool) -> dict[str, object]:
    attributes: dict[str, object] = {
        "label": _text(unit.title),
        "title": _text(unit.title),
        "source_project": _field_value(unit.source_project),
        "tags": _json_text(unit.tags),
    }
    if include_metadata:
        attributes["metadata"] = _json_text(unit.metadata)
    return attributes


def _edge_attributes(edge: KnowledgeEdge, *, include_metadata: bool) -> dict[str, object]:
    relation = _field_value(edge.relation)
    attributes: dict[str, object] = {
        "label": relation,
        "relation": relation,
        "weight": edge.weight,
    }
    if include_metadata:
        attributes["metadata"] = _json_text(edge.metadata)
    return attributes


def _attributes(attributes: Mapping[str, object]) -> str:
    return ", ".join(f"{key}={_dot_string(value)}" for key, value in attributes.items())


def _dot_string(value: object) -> str:
    text = _text(value)
    escaped = (
        text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    )
    return f'"{escaped}"'


def _json_text(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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
    return _text(getattr(value, "value", value))


def _text(value: object) -> str:
    return str(value or "")
