"""CSV export for temporal observation gaps per unit."""

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
    "unit_title",
    "source_project",
    "source_entity_type",
    "observation_count",
    "first_observed_date",
    "last_observed_date",
    "observed_span_days",
    "largest_gap_days",
    "has_multi_observation_gap",
]
_DATE_FIELDS = ("created_at", "updated_at", "ingested_at")
_METADATA_DATE_KEYS = ("observed_at", "observed_date", "source_date", "date", "published_at")
_METADATA_DATE_LIST_KEYS = ("observation_dates", "observed_dates", "source_dates", "dates")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_observation_gap_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one temporal observation-gap row per unit."""
    unit_list = list(units)
    rows = _gap_rows(unit_list)
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


def _gap_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        dates = _unit_dates(unit)
        first_observed = dates[0] if dates else None
        last_observed = dates[-1] if dates else None
        largest_gap = _largest_gap_days(dates)
        rows.append(
            {
                "unit_id": _field_value(unit.id),
                "unit_title": _field_value(unit.title),
                "source_project": _field_value(unit.source_project) or "Unknown",
                "source_entity_type": _field_value(unit.source_entity_type) or "Unknown",
                "observation_count": len(dates),
                "first_observed_date": first_observed.isoformat() if first_observed else "",
                "last_observed_date": last_observed.isoformat() if last_observed else "",
                "observed_span_days": (last_observed - first_observed).days
                if first_observed and last_observed
                else "",
                "largest_gap_days": largest_gap if largest_gap is not None else "",
                "has_multi_observation_gap": "true" if largest_gap is not None and largest_gap > 0 else "false",
            }
        )

    return sorted(rows, key=lambda row: (_sort_key(row["unit_id"]), _sort_key(row["unit_title"])))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_dates(unit: KnowledgeUnit) -> list[date]:
    values = [_date_value(getattr(unit, field, None)) for field in _DATE_FIELDS]
    metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
    values.extend(_date_value(metadata.get(key)) for key in _METADATA_DATE_KEYS)
    for key in _METADATA_DATE_LIST_KEYS:
        values.extend(_date_value(value) for value in _iter_values(metadata.get(key)))
    return sorted({value for value in values if value is not None})


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


def _largest_gap_days(values: list[date]) -> int | None:
    if len(values) < 2:
        return None
    return max((current - previous).days for previous, current in zip(values, values[1:]))


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
