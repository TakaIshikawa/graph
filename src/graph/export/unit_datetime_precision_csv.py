"""CSV export for per-unit datetime precision diagnostics."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_project",
    "source_entity_type",
    "field_name",
    "raw_value",
    "precision",
    "timezone_present",
]
_DATE_FIELDS = ("created_at", "updated_at")
_DATE_KEY_RE = re.compile(r"(date|time|timestamp|at|start|end|published)", re.IGNORECASE)
_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_YEAR_ONLY_RE = re.compile(r"^\d{4}$")
_TIMEZONE_RE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_datetime_precision_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write deterministic per-unit datetime precision rows."""
    unit_list = list(units)
    rows = _datetime_rows(unit_list)
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


def _datetime_rows(units: list[KnowledgeUnit]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for unit in units:
        for field_name in _DATE_FIELDS:
            value = getattr(unit, field_name, None)
            if value is not None:
                rows.append(_row(unit, field_name, value))

        metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
        for key, value in metadata.items():
            field_name = _inline_text(key)
            if not field_name or not _DATE_KEY_RE.search(field_name):
                continue
            for item in _iter_values(value):
                if _inline_text(item):
                    rows.append(_row(unit, f"metadata.{field_name}", item))

    return sorted(rows, key=lambda row: (_sort_key(row["unit_id"]), _sort_key(row["field_name"]), _sort_key(row["raw_value"])))


def _row(unit: KnowledgeUnit, field_name: str, value: object) -> dict[str, str]:
    raw_value = _raw_value(value)
    return {
        "unit_id": _unit_id(unit),
        "title": _inline_text(unit.title),
        "source_project": _unit_source(unit),
        "source_entity_type": _unit_source_type(unit),
        "field_name": field_name,
        "raw_value": raw_value,
        "precision": _precision(value),
        "timezone_present": "true" if _timezone_present(value, raw_value) else "false",
    }


def _iter_values(value: object) -> list[object]:
    if isinstance(value, list | tuple | set):
        return list(value)
    if value is None:
        return []
    return [value]


def _precision(value: object) -> str:
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"

    text = _inline_text(value)
    if _YEAR_ONLY_RE.fullmatch(text):
        return "year"
    if _YEAR_MONTH_RE.fullmatch(text):
        return "year_month"
    if _DATE_ONLY_RE.fullmatch(text):
        return "date"
    if _DATETIME_RE.fullmatch(text):
        return "datetime"
    return "unknown"


def _timezone_present(value: object, raw_value: str) -> bool:
    if isinstance(value, datetime):
        return value.tzinfo is not None and value.utcoffset() is not None
    return bool(_TIMEZONE_RE.search(raw_value))


def _raw_value(value: object) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    return _inline_text(value)


def _render_csv(rows: list[dict[str, str]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id or unit.source_id)


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_source_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or "Unknown"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
