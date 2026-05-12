"""CSV edge-list export helpers for graph interoperability."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import date, datetime
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

FIELDNAMES = [
    "source_id",
    "target_id",
    "relationship",
    "type",
    "label",
    "source_adapter",
    "weight",
    "created_at",
    "metadata_json",
]


def export_graph_edges_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one deterministic CSV row per graph edge."""
    edge_list = sorted(list(edges), key=_edge_key)
    text = _render_csv([_edge_row(edge) for edge in edge_list])

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "edges_exported": len(edge_list),
        "bytes_written": output_path.stat().st_size,
        "fieldnames": FIELDNAMES,
    }


def _edge_row(edge: KnowledgeEdge) -> dict[str, Any]:
    relationship = _field_value(edge.relation)
    return {
        "source_id": _text(edge.from_unit_id),
        "target_id": _text(edge.to_unit_id),
        "relationship": relationship,
        "type": relationship,
        "label": _edge_label(edge, relationship),
        "source_adapter": _field_value(edge.source),
        "weight": edge.weight,
        "created_at": _datetime_text(edge.created_at),
        "metadata_json": _json_text(edge.metadata),
    }


def _edge_label(edge: KnowledgeEdge, relationship: str) -> str:
    for key in ("label", "title", "name"):
        value = edge.metadata.get(key)
        if value:
            return _text(value)
    return relationship


def _render_csv(rows: list[dict[str, Any]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _json_text(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=_item_key)}
    if isinstance(value, list | tuple | set):
        return [_json_value(item) for item in value]
    return str(value)


def _edge_key(edge: KnowledgeEdge) -> tuple[str, str, str, str]:
    return (
        _text(edge.from_unit_id),
        _text(edge.to_unit_id),
        _field_value(edge.relation),
        _text(edge.id),
    )


def _datetime_text(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return _text(value)


def _field_value(value: object) -> str:
    return _text(getattr(value, "value", value))


def _item_key(item: tuple[Any, Any]) -> str:
    return str(item[0])


def _text(value: object) -> str:
    return str(value or "")
