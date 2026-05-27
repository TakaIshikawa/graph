"""Summarize ingest frequency by unit source."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import timezone
from typing import Any

from graph.export._report_csv import field_value, get, metadata, parse_datetime, sort_key

_SOURCE_KEYS = ("source", "source_project")
_TIMESTAMP_KEYS = ("ingested_at", "imported_at", "created_at")


def summarize_source_ingest_frequency(units: Iterable[Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    total = 0

    for unit in units:
        total += 1
        meta = metadata(unit)
        source = _source(unit, meta)
        group = groups.setdefault(
            source,
            {
                "source": source,
                "unit_count": 0,
                "dated_count": 0,
                "undated_count": 0,
                "dates": [],
            },
        )
        group["unit_count"] += 1

        ingest_date = _ingest_date(unit, meta)
        if ingest_date is None:
            group["undated_count"] += 1
            continue
        group["dated_count"] += 1
        group["dates"].append(ingest_date)

    rows = []
    for source in sorted(groups, key=sort_key):
        group = groups[source]
        active_dates = sorted(set(group["dates"]))
        active_day_count = len(active_dates)
        rows.append(
            {
                "source": source,
                "unit_count": group["unit_count"],
                "dated_count": group["dated_count"],
                "undated_count": group["undated_count"],
                "first_ingest_date": active_dates[0].isoformat() if active_dates else None,
                "last_ingest_date": active_dates[-1].isoformat() if active_dates else None,
                "active_day_count": active_day_count,
                "average_units_per_active_day": round(group["dated_count"] / active_day_count, 2)
                if active_day_count
                else 0.0,
            }
        )

    return {"total_units": total, "sources": rows}


def _source(unit: Any, meta: Mapping[str, Any]) -> str:
    for key in _SOURCE_KEYS:
        value = field_value(get(unit, key))
        if value:
            return value
        value = field_value(meta.get(key))
        if value:
            return value
    return "unknown"


def _ingest_date(unit: Any, meta: Mapping[str, Any]) -> Any:
    for key in _TIMESTAMP_KEYS:
        parsed = parse_datetime(get(unit, key))
        if parsed is None:
            parsed = parse_datetime(meta.get(key))
        if parsed is not None:
            return parsed.astimezone(timezone.utc).date()
    return None
