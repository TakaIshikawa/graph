"""CSV export for unit timestamp completeness."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, parse_datetime, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "created_at", "updated_at", "published_at", "accessed_at", "earliest_timestamp", "latest_timestamp", "completeness_score", "invalid_timestamp_fields"]
_TIMESTAMP_KEYS = ("created_at", "updated_at", "published_at", "accessed_at")


def export_units_to_timestamp_completeness_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str]:
    values: dict[str, str] = {}
    parsed: list[datetime] = []
    invalid: list[str] = []
    for key in _TIMESTAMP_KEYS:
        raw = get(unit, key) or metadata(unit).get(key)
        text = _timestamp_text(raw)
        values[key] = text
        if not text:
            continue
        when = parse_datetime(raw)
        if when is None:
            invalid.append(key)
        else:
            parsed.append(when)
    present_valid = sum(1 for key in _TIMESTAMP_KEYS if values[key] and key not in invalid)
    return {
        "unit_id": unit_id(unit),
        **values,
        "earliest_timestamp": _format_timestamp(min(parsed)) if parsed else "",
        "latest_timestamp": _format_timestamp(max(parsed)) if parsed else "",
        "completeness_score": f"{present_valid / len(_TIMESTAMP_KEYS):.2f}",
        "invalid_timestamp_fields": "; ".join(invalid),
    }


def _timestamp_text(value: object) -> str:
    parsed = parse_datetime(value)
    return _format_timestamp(parsed) if parsed else field_value(value)


def _format_timestamp(value: datetime) -> str:
    when = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return when.isoformat()
