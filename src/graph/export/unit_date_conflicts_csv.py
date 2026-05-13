"""CSV export for units with conflicting date fields."""

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
    "earliest_date",
    "latest_date",
    "span_days",
    "date_count",
    "date_fields",
]
_WHITESPACE_RE = re.compile(r"\s+")
_BUILT_IN_DATE_FIELDS = ("created_at", "updated_at", "ingested_at")
_METADATA_DATE_KEYS = ("date", "source_date", "published_at", "observed_at")


def export_unit_date_conflicts_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    minimum_span_days: int = 1,
) -> str | dict[str, Any]:
    """Return or write rows for units whose parsed dates span at least a threshold."""
    unit_list = list(units)
    rows = _conflict_rows(unit_list, minimum_span_days)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "conflict_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _conflict_rows(units: list[KnowledgeUnit], minimum_span_days: int) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        field_dates = _field_dates(unit)
        distinct_dates = sorted({value for _, value in field_dates})
        if len(distinct_dates) < 2:
            continue
        span_days = (distinct_dates[-1] - distinct_dates[0]).days
        if span_days < minimum_span_days:
            continue
        rows.append(
            {
                "unit_id": _field_value(getattr(unit, "id", None)),
                "source_project": _field_value(getattr(unit, "source_project", None)) or "Unknown",
                "source_entity_type": _field_value(getattr(unit, "source_entity_type", None)) or "Unknown",
                "earliest_date": distinct_dates[0].isoformat(),
                "latest_date": distinct_dates[-1].isoformat(),
                "span_days": span_days,
                "date_count": len(distinct_dates),
                "date_fields": "; ".join(f"{field}={value.isoformat()}" for field, value in field_dates),
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["source_entity_type"]), _sort_key(row["unit_id"])))


def _field_dates(unit: KnowledgeUnit) -> list[tuple[str, date]]:
    values: list[tuple[str, date]] = []
    for field in _BUILT_IN_DATE_FIELDS:
        parsed = _date_value(getattr(unit, field, None))
        if parsed is not None:
            values.append((field, parsed))
    metadata = getattr(unit, "metadata", None)
    if isinstance(metadata, dict):
        for key in _METADATA_DATE_KEYS:
            parsed = _date_value(metadata.get(key))
            if parsed is not None:
                values.append((f"metadata.{key}", parsed))
    return sorted(values, key=lambda item: (_sort_key(item[0]), item[1].isoformat()))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


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


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
