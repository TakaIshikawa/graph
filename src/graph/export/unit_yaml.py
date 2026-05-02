"""YAML export helpers for knowledge units."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from graph.types.models import KnowledgeUnit


def export_units_to_yaml(units: Iterable[KnowledgeUnit], path: str | Path) -> dict:
    """Write units to YAML with deterministic records and stats."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_units = list(units)
    exported_units = (
        all_units
        if isinstance(units, Sequence)
        else sorted(all_units, key=_unit_sort_key)
    )
    records = [_unit_record(unit) for unit in exported_units]
    text = yaml.safe_dump(
        records,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    output_path.write_text(text, encoding="utf-8")

    bytes_written = output_path.stat().st_size
    return {
        "path": str(output_path),
        "units_scanned": len(all_units),
        "units_exported": len(exported_units),
        "bytes_written": bytes_written,
    }


def _unit_record(unit: KnowledgeUnit) -> dict[str, Any]:
    return {
        "id": _yaml_value(unit.id),
        "source_project": _yaml_value(unit.source_project),
        "source_id": _yaml_value(unit.source_id),
        "source_entity_type": _yaml_value(unit.source_entity_type),
        "title": _yaml_value(unit.title),
        "content": _yaml_value(unit.content),
        "content_type": _yaml_value(unit.content_type),
        "tags": _yaml_value(unit.tags),
        "confidence": _yaml_value(unit.confidence),
        "utility_score": _yaml_value(unit.utility_score),
        "created_at": _yaml_value(unit.created_at),
        "ingested_at": _yaml_value(unit.ingested_at),
        "updated_at": _yaml_value(unit.updated_at),
        "metadata": _yaml_value(unit.metadata),
    }


def _yaml_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _yaml_value(value.model_dump())
    if isinstance(value, Mapping):
        return {
            str(key): _yaml_value(item)
            for key, item in sorted(value.items(), key=_item_key)
        }
    if isinstance(value, list | tuple):
        return [_yaml_value(item) for item in value]
    return str(value)


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (
        str(_yaml_value(unit.source_project) or ""),
        str(_yaml_value(unit.source_id) or ""),
        str(_yaml_value(unit.title) or ""),
    )


def _item_key(item: tuple[Any, Any]) -> str:
    return str(item[0])
