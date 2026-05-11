"""Neo4j admin-import compatible CSV export helpers."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from io import StringIO
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_NODE_HEADERS = [":ID", ":LABEL", "title", "content_type", "metadata", "tags"]
_REL_HEADERS = [":START_ID", ":END_ID", ":TYPE", "weight", "metadata"]


def export_graph_neo4j_bulk_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
) -> dict[str, str]:
    """Return deterministic Neo4j admin-import CSV contents."""
    node_rows = [_node_row(unit) for unit in sorted(list(units), key=_unit_key)]
    rel_rows = [_relationship_row(edge) for edge in sorted(list(edges), key=_edge_key)]
    return {
        "nodes.csv": _render_csv(node_rows, _NODE_HEADERS),
        "relationships.csv": _render_csv(rel_rows, _REL_HEADERS),
    }


def _node_row(unit: KnowledgeUnit) -> dict[str, Any]:
    labels = ["KnowledgeUnit"]
    source_project = _field_value(unit.source_project)
    entity_type = _text(unit.source_entity_type)
    for label in (source_project, entity_type):
        normalized = _neo4j_label(label)
        if normalized and normalized not in labels:
            labels.append(normalized)
    return {
        ":ID": _unit_id(unit),
        ":LABEL": ";".join(labels),
        "title": _text(unit.title),
        "content_type": _field_value(unit.content_type),
        "metadata": _json_text(unit.metadata),
        "tags": _json_text(sorted({_text(tag) for tag in unit.tags if _text(tag)}, key=_sort_key)),
    }


def _relationship_row(edge: KnowledgeEdge) -> dict[str, Any]:
    return {
        ":START_ID": _text(edge.from_unit_id),
        ":END_ID": _text(edge.to_unit_id),
        ":TYPE": _relationship_type(edge.relation),
        "weight": edge.weight,
        "metadata": _json_text(edge.metadata),
    }


def _render_csv(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _relationship_type(value: object) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", _field_value(value).upper()).strip("_")
    return normalized or "RELATES_TO"


def _neo4j_label(value: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", value)
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


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
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list | tuple | set):
        return [_json_value(item) for item in value]
    return str(value)


def _unit_key(unit: KnowledgeUnit) -> tuple[str, str]:
    return (_unit_id(unit), _text(unit.source_id))


def _edge_key(edge: KnowledgeEdge) -> tuple[str, str, str, str]:
    return (_text(edge.from_unit_id), _text(edge.to_unit_id), _relationship_type(edge.relation), _text(edge.id))


def _unit_id(unit: KnowledgeUnit) -> str:
    return _text(unit.id or unit.source_id)


def _field_value(value: object) -> str:
    return _text(getattr(value, "value", value))


def _sort_key(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)


def _text(value: object) -> str:
    return str(value or "")
