"""JSON export helpers for knowledge units."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit


def export_units_to_json(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    metadata_keys: list[str] | None = None,
) -> str:
    """Serialize units as deterministic JSON.

    Args:
        units: Single unit or iterable of units to export
        path: Optional file path to write JSON output
        metadata_keys: Optional list of metadata keys to include (if None, include all)

    Returns:
        JSON string representation of the units
    """
    if isinstance(units, KnowledgeUnit):
        record = _unit_record(units, metadata_keys=metadata_keys)
        text = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2)
    else:
        records = [_unit_record(unit, metadata_keys=metadata_keys) for unit in units]
        text = json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2)

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    return text


def _unit_record(unit: KnowledgeUnit, *, metadata_keys: list[str] | None) -> dict[str, Any]:
    metadata = _json_value(unit.metadata)
    if metadata_keys is not None:
        metadata = {key: metadata[key] for key in metadata_keys if key in metadata}

    return {
        "id": unit.id,
        "source_project": _json_value(unit.source_project),
        "source_id": unit.source_id,
        "source_entity_type": unit.source_entity_type,
        "title": unit.title,
        "content": unit.content,
        "content_type": _json_value(unit.content_type),
        "metadata": metadata,
        "tags": _json_value(unit.tags),
        "confidence": unit.confidence,
        "utility_score": unit.utility_score,
        "created_at": _json_value(unit.created_at),
        "ingested_at": _json_value(unit.ingested_at),
        "updated_at": _json_value(unit.updated_at),
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
