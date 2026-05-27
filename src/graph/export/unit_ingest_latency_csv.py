"""CSV export for unit ingest latency."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, parse_datetime, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "source", "entity_type", "source_created_at", "ingested_at", "latency_hours", "latency_bucket"]
_SOURCE_CREATED_KEYS = ("source_created_at", "created_at", "published_at", "date_created", "created", "post_date", "date")


def export_units_to_ingest_latency_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((row for unit in unit_list if (row := _row(unit))), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str] | None:
    source_created_at = _source_created_at(unit)
    ingested_at = parse_datetime(get(unit, "ingested_at") or metadata(unit).get("ingested_at"))
    if source_created_at is None or ingested_at is None:
        return None
    hours = (ingested_at - source_created_at).total_seconds() / 3600
    return {
        "unit_id": unit_id(unit),
        "source": field_value(get(unit, "source_project") or metadata(unit).get("source")),
        "entity_type": field_value(get(unit, "source_entity_type") or metadata(unit).get("entity_type")),
        "source_created_at": source_created_at.isoformat(),
        "ingested_at": ingested_at.isoformat(),
        "latency_hours": f"{hours:.2f}",
        "latency_bucket": _bucket(hours),
    }


def _source_created_at(unit: Mapping[str, Any] | object) -> datetime | None:
    meta = metadata(unit)
    for key in _SOURCE_CREATED_KEYS:
        parsed = parse_datetime(meta.get(key) if key != "created_at" else (meta.get(key) or get(unit, key)))
        if parsed:
            return parsed
    return None


def _bucket(hours: float) -> str:
    if hours < 0:
        return "negative"
    if hours < 1:
        return "under_1h"
    if hours < 24:
        return "1h_24h"
    if hours < 168:
        return "1d_7d"
    return "7d_plus"
