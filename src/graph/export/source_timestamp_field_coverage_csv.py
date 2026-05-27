"""CSV export for source timestamp field coverage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, parse_datetime, render_csv, sort_key, write_csv

_DEFAULT_FIELDS = ("created_at", "updated_at", "published_at", "archived_at")
_FIELDNAMES = ["source_project", "field", "unit_count", "present_count", "coverage_ratio", "invalid_count"]
_UNKNOWN_SOURCE = "Unknown"


def export_source_timestamp_field_coverage_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    fields: Iterable[str] | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    field_list = [field_value(field) for field in (fields or _DEFAULT_FIELDS) if field_value(field)]
    rows = _rows(unit_list, field_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(units: list[Mapping[str, Any] | object], fields: list[str]) -> list[dict[str, str | int]]:
    groups: dict[str, list[Mapping[str, Any] | object]] = defaultdict(list)
    for unit in units:
        groups[_source(unit)].append(unit)
    rows = []
    for source in sorted(groups, key=sort_key):
        source_units = groups[source]
        for field in fields:
            present = invalid = 0
            for unit in source_units:
                value = _field(unit, field)
                if not field_value(value):
                    continue
                present += 1
                if parse_datetime(value) is None:
                    invalid += 1
            unit_count = len(source_units)
            rows.append(
                {
                    "source_project": source,
                    "field": field,
                    "unit_count": unit_count,
                    "present_count": present,
                    "coverage_ratio": f"{present / unit_count:.2f}",
                    "invalid_count": invalid,
                }
            )
    return rows


def _source(unit: Mapping[str, Any] | object) -> str:
    return field_value(get(unit, "source_project")) or _UNKNOWN_SOURCE


def _field(unit: Mapping[str, Any] | object, field: str) -> object:
    value = get(unit, field)
    return value if field_value(value) else metadata(unit).get(field)
