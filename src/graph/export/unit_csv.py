"""CSV export helpers for knowledge units."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit

_BASE_FIELDNAMES = [
    "id",
    "source_project",
    "source_id",
    "source_entity_type",
    "title",
    "content_type",
    "tags",
    "confidence",
    "utility_score",
    "created_at",
    "updated_at",
]
_CONTENT_FIELDNAME = "content"


def export_units_to_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path,
    *,
    include_content: bool = True,
    columns: Sequence[str] | None = None,
    metadata_fields: Sequence[str] | None = None,
) -> dict:
    """Write units to CSV with stable headers and configurable columns."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_units = list(units)
    exported_units = sorted(all_units, key=lambda unit: _text(unit.id))
    fieldnames = _fieldnames(
        exported_units,
        include_content=include_content,
        columns=columns,
        metadata_fields=metadata_fields,
    )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for unit in exported_units:
            row = _unit_row(unit, include_content=include_content)
            row.update(_metadata_row(unit.metadata, metadata_fields=metadata_fields))
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    bytes_written = output_path.stat().st_size
    stats = {
        "path": str(output_path),
        "units_scanned": len(all_units),
        "units_exported": len(exported_units),
        "content_included": include_content,
        "bytes_written": bytes_written,
    }
    if columns is not None or metadata_fields is not None:
        stats["columns"] = fieldnames
    return stats


def _fieldnames(
    units: list[KnowledgeUnit],
    *,
    include_content: bool,
    columns: Sequence[str] | None,
    metadata_fields: Sequence[str] | None,
) -> list[str]:
    if columns is not None:
        return list(columns)
    fieldnames = [*_BASE_FIELDNAMES]
    if include_content:
        fieldnames.append(_CONTENT_FIELDNAME)
    metadata_keys = list(metadata_fields or [])
    fieldnames.extend(f"metadata.{key}" for key in metadata_keys)
    return fieldnames


def _unit_row(unit: KnowledgeUnit, *, include_content: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": _text(unit.id),
        "source_project": _enum_value(unit.source_project),
        "source_id": _text(unit.source_id),
        "source_entity_type": _text(unit.source_entity_type),
        "title": _text(unit.title),
        "content_type": _enum_value(unit.content_type),
        "tags": _tags_text(unit.tags),
        "confidence": unit.confidence,
        "utility_score": unit.utility_score,
        "created_at": _datetime_text(unit.created_at),
        "updated_at": _datetime_text(unit.updated_at),
    }
    if include_content:
        row[_CONTENT_FIELDNAME] = _text(unit.content)
    return row


def _tags_text(tags: Iterable[object]) -> str:
    return ";".join(sorted(_text(tag) for tag in tags))


def _enum_value(value: object) -> str:
    return _text(getattr(value, "value", value))


def _datetime_text(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return _text(value)


def _metadata_row(metadata: Mapping[str, Any], *, metadata_fields: Sequence[str] | None) -> dict[str, Any]:
    flattened = _flatten_metadata(metadata)
    keys = list(metadata_fields) if metadata_fields is not None else list(flattened)
    return {f"metadata.{key}": _csv_value(flattened.get(key, "")) for key in keys}


def _flatten_metadata(metadata: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in sorted(metadata.items(), key=lambda item: str(item[0])):
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_metadata(value, dotted))
        else:
            flattened[dotted] = value
    return flattened


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _csv_value(value.model_dump())
    if isinstance(value, Mapping):
        return "; ".join(
            f"{key}: {_csv_value(item)}"
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if _csv_value(item)
        )
    if isinstance(value, list | tuple | set):
        return ";".join(_csv_value(item) for item in value)
    return str(value)


def _text(value: object) -> str:
    return str(value or "")
