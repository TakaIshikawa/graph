"""JSON Lines export helpers for knowledge units."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit


def export_units_to_jsonl(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    include_embedding: bool = False,
) -> str:
    """Serialize units as deterministic newline-delimited JSON."""
    lines = [
        json.dumps(
            _unit_record(unit, include_embedding=include_embedding),
            ensure_ascii=False,
            sort_keys=True,
        )
        for unit in units
    ]
    text = "".join(f"{line}\n" for line in lines)

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    return text


def _unit_record(unit: KnowledgeUnit, *, include_embedding: bool) -> dict[str, Any]:
    record = {
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
    if include_embedding:
        record["embedding"] = _json_value(unit.embedding)
    return record


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
