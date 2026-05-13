"""CSV export for per-unit lifecycle date spans."""

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
    "source_project",
    "source_entity_type",
    "created_date",
    "ingested_date",
    "updated_date",
    "created_to_ingested_days",
    "created_to_updated_days",
    "ingested_to_updated_days",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_date_span_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write deterministic per-unit lifecycle date spans."""
    unit_list = sorted(list(units), key=_unit_sort_key)
    rows = [_row(unit) for unit in unit_list]
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


def _row(unit: KnowledgeUnit) -> dict[str, str | int]:
    created_date = _date_value(getattr(unit, "created_at", None))
    ingested_date = _date_value(getattr(unit, "ingested_at", None))
    updated_date = _date_value(getattr(unit, "updated_at", None))

    return {
        "unit_id": _unit_id(unit),
        "source_project": _unit_source(unit),
        "source_entity_type": _unit_source_type(unit),
        "created_date": created_date.isoformat() if created_date else "",
        "ingested_date": ingested_date.isoformat() if ingested_date else "",
        "updated_date": updated_date.isoformat() if updated_date else "",
        "created_to_ingested_days": _span_days(created_date, ingested_date),
        "created_to_updated_days": _span_days(created_date, updated_date),
        "ingested_to_updated_days": _span_days(ingested_date, updated_date),
    }


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


def _span_days(start: date | None, end: date | None) -> int | str:
    if start is None or end is None:
        return ""
    return (end - start).days


def _render_csv(rows: list[dict[str, str | int]]) -> str:
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


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    return (_sort_key(_unit_source(unit)), _sort_key(_unit_source_type(unit)), _sort_key(_unit_id(unit)))
