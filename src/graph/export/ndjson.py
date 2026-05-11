"""NDJSON export helpers for knowledge graph records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from graph.types.models import KnowledgeEdge, KnowledgeUnit

RecordType = Literal["both", "units", "edges"]


def export_graph_ndjson(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge] | None = None,
    path: str | Path | None = None,
    *,
    record_type: RecordType = "both",
) -> str:
    """Serialize graph units and edges as compact newline-delimited JSON."""
    if record_type not in {"both", "units", "edges"}:
        raise ValueError("record_type must be one of: both, units, edges")

    unit_list = [units] if isinstance(units, KnowledgeUnit) else list(units)
    edge_list = list(edges or [])

    records: list[dict[str, Any]] = []
    if record_type in {"both", "units"}:
        records.extend(_unit_record(unit) for unit in unit_list)
    if record_type in {"both", "edges"}:
        records.extend(_edge_record(edge) for edge in edge_list)

    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in sorted(records, key=_record_key)
    ]
    text = "".join(f"{line}\n" for line in lines)

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    return text


def export_units_to_ndjson(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str:
    """Serialize units as compact newline-delimited JSON records."""
    return export_graph_ndjson(units, [], path, record_type="units")


def _unit_record(unit: KnowledgeUnit) -> dict[str, Any]:
    return {
        "record_type": "unit",
        "id": unit.id,
        "source_project": _json_value(unit.source_project),
        "source_id": unit.source_id,
        "source_entity_type": unit.source_entity_type,
        "title": unit.title,
        "content": unit.content,
        "content_type": _json_value(unit.content_type),
        "metadata": _json_value(unit.metadata),
        "tags": _json_value(unit.tags),
        "confidence": unit.confidence,
        "utility_score": unit.utility_score,
        "created_at": _json_value(unit.created_at),
        "ingested_at": _json_value(unit.ingested_at),
        "updated_at": _json_value(unit.updated_at),
    }


def _edge_record(edge: KnowledgeEdge) -> dict[str, Any]:
    return {
        "record_type": "edge",
        "id": edge.id,
        "from_unit_id": edge.from_unit_id,
        "to_unit_id": edge.to_unit_id,
        "relation": _json_value(edge.relation),
        "weight": edge.weight,
        "source": _json_value(edge.source),
        "metadata": _json_value(edge.metadata),
        "created_at": _json_value(edge.created_at),
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
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    return str(value)


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("created_at") or ""),
        str(record.get("record_type") or ""),
        str(record.get("id") or ""),
    )


def _item_key(item: tuple[Any, Any]) -> str:
    return str(item[0])
