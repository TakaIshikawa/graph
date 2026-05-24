"""Store-level ingest latency summaries for knowledge units."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

DEFAULT_BUCKETS_SECONDS = (60, 300, 900, 3600)
DEFAULT_SOURCE_TIMESTAMP_KEYS = (
    "source_fetched_at",
    "fetched_at",
    "fetch_at",
    "source_imported_at",
    "imported_at",
    "import_at",
    "crawled_at",
    "scraped_at",
    "downloaded_at",
    "retrieved_at",
)
DEFAULT_UNIT_TIMESTAMP_FIELDS = ("ingested_at", "created_at", "embedding_updated_at", "updated_at")


def summarize_unit_ingest_latency(
    store: Any,
    *,
    bucket_bounds_seconds: Sequence[int | float] = DEFAULT_BUCKETS_SECONDS,
    source_timestamp_keys: Sequence[str] = DEFAULT_SOURCE_TIMESTAMP_KEYS,
    unit_timestamp_fields: Sequence[str] = DEFAULT_UNIT_TIMESTAMP_FIELDS,
) -> dict[str, Any]:
    """Compute latency from source fetch/import timestamps to unit ingest timestamps.

    ``store`` may be a ``Store`` instance or a ``sqlite3.Connection``. Source timestamps are
    read from unit metadata; unit timestamps are read from configured ``knowledge_units`` fields.
    """

    bounds = _validate_bucket_bounds(bucket_bounds_seconds)
    source_keys = _validate_names(source_timestamp_keys, "source_timestamp_keys")
    unit_fields = _validate_unit_timestamp_fields(unit_timestamp_fields)
    conn = _connection_from_store(store)

    rows = conn.execute(
        """
        SELECT id, source_project, source_id, source_entity_type, title, metadata,
               created_at, ingested_at, updated_at, embedding_updated_at
        FROM knowledge_units
        ORDER BY source_project, source_id, source_entity_type, id
        """
    ).fetchall()

    latency_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []
    latencies: list[float] = []

    for row in rows:
        metadata = _load_metadata(row["metadata"])
        source_timestamp = _first_metadata_timestamp(metadata, source_keys)
        unit_timestamp = _first_row_timestamp(row, unit_fields)
        row_ref = {
            "unit_id": str(row["id"]),
            "source_project": str(row["source_project"]),
            "source_id": str(row["source_id"]),
            "source_entity_type": str(row["source_entity_type"]),
        }

        if source_timestamp is None:
            skipped_rows.append({**row_ref, "reason": "missing_source_timestamp"})
            continue
        if unit_timestamp is None:
            skipped_rows.append({**row_ref, "reason": "missing_unit_timestamp"})
            continue

        latency_seconds = (unit_timestamp["value"] - source_timestamp["value"]).total_seconds()
        if latency_seconds < 0:
            skipped_rows.append({**row_ref, "reason": "negative_latency"})
            continue

        latencies.append(latency_seconds)
        latency_rows.append(
            {
                **row_ref,
                "source_timestamp_key": source_timestamp["key"],
                "source_timestamp": source_timestamp["value"].isoformat(),
                "unit_timestamp_field": unit_timestamp["key"],
                "unit_timestamp": unit_timestamp["value"].isoformat(),
                "latency_seconds": latency_seconds,
                "bucket": _bucket_label(latency_seconds, bounds),
            }
        )

    return {
        "count": len(latencies),
        "min_seconds": min(latencies) if latencies else None,
        "max_seconds": max(latencies) if latencies else None,
        "average_seconds": (sum(latencies) / len(latencies)) if latencies else None,
        "buckets": _bucket_rows(latencies, bounds),
        "latency_rows": latency_rows,
        "skipped_rows": skipped_rows,
        "skipped_count": len(skipped_rows),
    }


def _connection_from_store(store: Any) -> sqlite3.Connection:
    conn = getattr(store, "conn", store)
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("store must be a Store or sqlite3.Connection")
    conn.row_factory = sqlite3.Row
    return conn


def _validate_bucket_bounds(bounds: Sequence[int | float]) -> tuple[float, ...]:
    if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Sequence):
        raise ValueError("bucket_bounds_seconds must be a sequence of non-negative numbers")
    normalized: list[float] = []
    previous: float | None = None
    for bound in bounds:
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or bound < 0:
            raise ValueError("bucket_bounds_seconds must be a sequence of non-negative numbers")
        value = float(bound)
        if previous is not None and value <= previous:
            raise ValueError("bucket_bounds_seconds must be strictly increasing")
        normalized.append(value)
        previous = value
    return tuple(normalized)


def _validate_names(names: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise ValueError(f"{label} must be a sequence of non-empty strings")
    normalized = tuple(name for name in names if isinstance(name, str) and name.strip())
    if len(normalized) != len(names):
        raise ValueError(f"{label} must be a sequence of non-empty strings")
    return normalized


def _validate_unit_timestamp_fields(fields: Sequence[str]) -> tuple[str, ...]:
    allowed = {"created_at", "ingested_at", "updated_at", "embedding_updated_at"}
    normalized = _validate_names(fields, "unit_timestamp_fields")
    invalid = sorted(set(normalized) - allowed)
    if invalid:
        raise ValueError("unit_timestamp_fields contains unsupported fields: " + ", ".join(invalid))
    return normalized


def _load_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _first_metadata_timestamp(
    metadata: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any] | None:
    for key in keys:
        timestamp = _parse_datetime(_metadata_value(metadata, key))
        if timestamp is not None:
            return {"key": key, "value": timestamp}
    return None


def _first_row_timestamp(row: sqlite3.Row, fields: tuple[str, ...]) -> dict[str, Any] | None:
    for field in fields:
        timestamp = _parse_datetime(row[field])
        if timestamp is not None:
            return {"key": field, "value": timestamp}
    return None


def _metadata_value(metadata: dict[str, Any], path: str) -> Any:
    current: Any = metadata
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


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


def _bucket_label(value: float, bounds: tuple[float, ...]) -> str:
    previous = 0.0
    for bound in bounds:
        if value <= bound:
            return f"{_format_bound(previous)}-{_format_bound(bound)}"
        previous = bound
    return f">{_format_bound(bounds[-1])}" if bounds else "all"


def _bucket_rows(values: list[float], bounds: tuple[float, ...]) -> list[dict[str, Any]]:
    labels = [_bucket_label(bound, bounds) for bound in bounds]
    if bounds:
        labels.append(f">{_format_bound(bounds[-1])}")
    else:
        labels.append("all")
    counts = {label: 0 for label in labels}
    for value in values:
        counts[_bucket_label(value, bounds)] += 1
    return [{"bucket": label, "count": counts[label]} for label in labels]


def _format_bound(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)
