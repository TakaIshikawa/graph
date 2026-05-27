"""Summarize lag between source timestamps and graph ingest timestamps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

DEFAULT_SAMPLE_LIMIT = 5
_SOURCE_TIMESTAMP_KEYS = ("source_updated_at", "source_created_at", "published_at")
_GRAPH_TIMESTAMP_KEYS = ("ingested_at", "created_at")
_SOURCE_KEYS = ("source", "source_project")


def unit_source_lag_summary(
    units: Iterable[Any],
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> list[dict[str, Any]]:
    """Return source-level lag rows in hours for KnowledgeUnit-like values."""

    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

    groups: dict[str, dict[str, Any]] = {}
    for unit in units:
        metadata = _metadata(unit)
        source = _source(unit, metadata)
        unit_id = _text(_get(unit, "id") or metadata.get("id"))
        group = groups.setdefault(
            source,
            {
                "source": source,
                "count": 0,
                "lags": [],
                "negative_lag_count": 0,
                "missing_timestamp_count": 0,
                "sample_unit_ids": [],
            },
        )
        group["count"] += 1
        if unit_id and len(group["sample_unit_ids"]) < sample_limit:
            group["sample_unit_ids"].append(unit_id)

        source_timestamp = _first_timestamp(unit, metadata, _SOURCE_TIMESTAMP_KEYS)
        graph_timestamp = _first_timestamp(unit, metadata, _GRAPH_TIMESTAMP_KEYS)
        if source_timestamp is None or graph_timestamp is None:
            group["missing_timestamp_count"] += 1
            continue

        lag_hours = (graph_timestamp - source_timestamp).total_seconds() / 3600
        if lag_hours < 0:
            group["negative_lag_count"] += 1
            continue
        group["lags"].append(lag_hours)

    rows: list[dict[str, Any]] = []
    for source in sorted(groups, key=_sort_key):
        group = groups[source]
        lags = group["lags"]
        rows.append(
            {
                "source": source,
                "count": group["count"],
                "average_lag_hours": sum(lags) / len(lags) if lags else None,
                "max_lag_hours": max(lags) if lags else None,
                "negative_lag_count": group["negative_lag_count"],
                "missing_timestamp_count": group["missing_timestamp_count"],
                "sample_unit_ids": group["sample_unit_ids"],
            }
        )
    return rows


def _first_timestamp(
    unit: Any, metadata: Mapping[str, Any], keys: tuple[str, ...]
) -> datetime | None:
    for key in keys:
        parsed = _parse_datetime(_get(unit, key))
        if parsed is not None:
            return parsed
        parsed = _parse_datetime(metadata.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source(unit: Any, metadata: Mapping[str, Any]) -> str:
    for key in _SOURCE_KEYS:
        value = _get(unit, key)
        if value not in (None, ""):
            return _text(value) or "unknown"
        value = metadata.get(key)
        if value not in (None, ""):
            return _text(value) or "unknown"
    return "unknown"


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _sort_key(value: str) -> tuple[str, str]:
    return (value.casefold(), value)
