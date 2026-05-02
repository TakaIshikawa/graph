"""CSV export helpers for knowledge units."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

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
) -> dict:
    """Write units to CSV with stable headers and deterministic row ordering."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_units = list(units)
    exported_units = sorted(all_units, key=lambda unit: _text(unit.id))
    fieldnames = [*_BASE_FIELDNAMES]
    if include_content:
        fieldnames.append(_CONTENT_FIELDNAME)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for unit in exported_units:
            writer.writerow(_unit_row(unit, include_content=include_content))

    bytes_written = output_path.stat().st_size
    return {
        "path": str(output_path),
        "units_scanned": len(all_units),
        "units_exported": len(exported_units),
        "content_included": include_content,
        "bytes_written": bytes_written,
    }


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


def _text(value: object) -> str:
    return str(value or "")
