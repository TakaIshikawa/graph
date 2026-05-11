"""Paired node and edge CSV export helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_NODE_FIELDNAMES = [
    "Id",
    "Label",
    "source_project",
    "source_entity_type",
    "content_type",
    "tags",
    "created_at",
    "updated_at",
    "metadata",
]
_EDGE_FIELDNAMES = ["Source", "Target", "Type", "Weight", "created_at", "metadata"]


def export_graph_node_edge_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    nodes_path: str | Path | None = None,
    edges_path: str | Path | None = None,
) -> tuple[str, str]:
    """Return deterministic node and edge CSV strings, optionally writing each file."""
    node_rows = [_node_row(unit) for unit in sorted(list(units), key=_unit_key)]
    edge_rows = [_edge_row(edge) for edge in sorted(list(edges), key=_edge_key)]
    nodes_csv = _render_csv(node_rows, _NODE_FIELDNAMES)
    edges_csv = _render_csv(edge_rows, _EDGE_FIELDNAMES)

    if nodes_path is not None:
        _write_text(nodes_path, nodes_csv)
    if edges_path is not None:
        _write_text(edges_path, edges_csv)

    return nodes_csv, edges_csv


def _node_row(unit: KnowledgeUnit) -> dict[str, Any]:
    return {
        "Id": _unit_id(unit),
        "Label": _text(unit.title),
        "source_project": _field_value(unit.source_project),
        "source_entity_type": _text(unit.source_entity_type),
        "content_type": _field_value(unit.content_type),
        "tags": ";".join(sorted({_text(tag) for tag in unit.tags if _text(tag)}, key=_sort_key)),
        "created_at": _datetime_text(unit.created_at),
        "updated_at": _datetime_text(unit.updated_at),
        "metadata": _json_text(unit.metadata),
    }


def _edge_row(edge: KnowledgeEdge) -> dict[str, Any]:
    return {
        "Source": _text(edge.from_unit_id),
        "Target": _text(edge.to_unit_id),
        "Type": _field_value(edge.relation),
        "Weight": edge.weight,
        "created_at": _datetime_text(edge.created_at),
        "metadata": _json_text(edge.metadata),
    }


def _render_csv(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")


def _json_text(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _unit_key(unit: KnowledgeUnit) -> tuple[str, str]:
    return (_unit_id(unit), _text(unit.source_id))


def _edge_key(edge: KnowledgeEdge) -> tuple[str, str, str, str]:
    return (
        _text(edge.from_unit_id),
        _text(edge.to_unit_id),
        _field_value(edge.relation),
        _text(edge.id),
    )


def _unit_id(unit: KnowledgeUnit) -> str:
    return _text(unit.id or unit.source_id)


def _field_value(value: object) -> str:
    return _text(getattr(value, "value", value))


def _datetime_text(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return _text(value)


def _item_key(item: tuple[Any, Any]) -> str:
    return str(item[0])


def _sort_key(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)


def _text(value: object) -> str:
    return str(value or "")
