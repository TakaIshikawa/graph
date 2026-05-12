"""CSV export for unit source timeline observations."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "unit_type",
    "source_id",
    "source_project",
    "source_entity_type",
    "observation_date",
    "confidence",
    "metadata_key_count",
]
_DATE_FIELDS = ("created_at", "ingested_at", "updated_at")
_METADATA_DATE_KEYS = ("observed_at", "observed_date", "source_date", "date", "published_at")
_METADATA_DATE_LIST_KEYS = ("observation_dates", "observed_dates", "source_dates", "dates")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_source_timeline_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per unit/source/date observation."""
    unit_list = list(units)
    rows = _timeline_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _timeline_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        dates = _unit_dates(unit)
        if not dates:
            dates = [""]
        for observation_date in dates:
            rows.append(
                {
                    "unit_id": _field_value(unit.id),
                    "unit_type": _field_value(unit.content_type),
                    "source_id": _field_value(unit.source_id),
                    "source_project": _field_value(unit.source_project) or "Unknown",
                    "source_entity_type": _field_value(unit.source_entity_type) or "Unknown",
                    "observation_date": observation_date,
                    "confidence": _decimal(unit.confidence) if _confidence_value(unit.confidence) is not None else "",
                    "metadata_key_count": _metadata_key_count(unit),
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["unit_id"]),
            _sort_key(row["source_project"]),
            _sort_key(row["source_entity_type"]),
            _sort_key(row["source_id"]),
            _sort_key(row["observation_date"]),
        ),
    )


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_dates(unit: KnowledgeUnit) -> list[str]:
    values = [_date_value(getattr(unit, field, None)) for field in _DATE_FIELDS]
    metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
    values.extend(_date_value(metadata.get(key)) for key in _METADATA_DATE_KEYS)
    for key in _METADATA_DATE_LIST_KEYS:
        values.extend(_date_value(value) for value in _iter_values(metadata.get(key)))
    return sorted({value.isoformat() for value in values if value is not None})


def _iter_values(value: object) -> list[object]:
    if isinstance(value, list | tuple | set):
        return list(value)
    if value is None:
        return []
    return [value]


def _date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _inline_text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _metadata_key_count(unit: KnowledgeUnit) -> int:
    metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
    return len([key for key in metadata if _field_value(key)])


def _confidence_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _decimal(value: float) -> str:
    return f"{value:.2f}"
