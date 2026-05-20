"""CSV export for parsed unit date ranges."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["source", "unit_id", "title", "date_count", "earliest_date", "latest_date", "span_days", "date_keys"]
_DATE_METADATA_KEYS = (
    "date",
    "source_date",
    "published_at",
    "published_date",
    "created_at",
    "updated_at",
    "ingested_at",
    "observed_at",
    "observed_date",
)
_DATE_FIELDS = ("created_at", "updated_at", "ingested_at")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_date_range_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write parsed date range coverage per unit."""
    unit_list = list(units)
    rows = _range_rows(unit_list)
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


def _range_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        unit_dates, keys = _unit_dates(unit)
        if not unit_dates:
            continue
        ordered_dates = sorted(unit_dates)
        rows.append(
            {
                "source": _field_value(_get(unit, "source_project")) or "Unknown",
                "unit_id": _unit_id(unit),
                "title": _field_value(_get(unit, "title")),
                "date_count": len(unit_dates),
                "earliest_date": ordered_dates[0].isoformat(),
                "latest_date": ordered_dates[-1].isoformat(),
                "span_days": (ordered_dates[-1] - ordered_dates[0]).days,
                "date_keys": "; ".join(sorted(keys, key=_sort_key)),
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["source"]), _sort_key(row["unit_id"])))


def _unit_dates(unit: KnowledgeUnit | Mapping[str, Any]) -> tuple[list[date], set[str]]:
    dates: list[date] = []
    keys: set[str] = set()
    metadata = _metadata(unit)
    for key in _DATE_METADATA_KEYS:
        parsed_values = _date_values(_casefold_get(metadata, key))
        if parsed_values:
            keys.add(key)
            dates.extend(parsed_values)
    for field in _DATE_FIELDS:
        parsed_values = _date_values(_get(unit, field))
        if parsed_values:
            keys.add(field)
            dates.extend(parsed_values)
    return dates, keys


def _date_values(value: object) -> list[date]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return []
    if isinstance(value, list | tuple | set):
        return [parsed for item in value for parsed in _date_values(item)]
    parsed = _date_value(value)
    return [parsed] if parsed is not None else []


def _date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _field_value(value)
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


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
