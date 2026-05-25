"""Store-level ingest timestamp daypart summaries."""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

DEFAULT_TIMESTAMP_FIELDS = ("ingested_at", "created_at")
ALLOWED_TIMESTAMP_FIELDS = {"created_at", "ingested_at", "updated_at", "embedding_updated_at"}


def summarize_unit_ingest_daypart(
    store: Any,
    *,
    timestamp_fields: Sequence[str] = DEFAULT_TIMESTAMP_FIELDS,
    timezone_offset_hours: int | float = 0,
) -> dict[str, Any]:
    """Bucket units by local weekday, hour, and daypart using ingest timestamps."""

    fields = _validate_unit_timestamp_fields(timestamp_fields)
    if isinstance(timezone_offset_hours, bool) or not isinstance(timezone_offset_hours, (int, float)):
        raise ValueError("timezone_offset_hours must be a number")
    offset = timedelta(hours=float(timezone_offset_hours))
    conn = _connection_from_store(store)
    rows = conn.execute(
        """
        SELECT id, source_project, content_type, created_at, ingested_at, updated_at, embedding_updated_at
        FROM knowledge_units
        ORDER BY id
        """
    ).fetchall()

    groups: dict[tuple[int, int, str], dict[str, Any]] = {}
    skipped_rows: list[dict[str, str]] = []
    for row in rows:
        timestamp = _first_row_timestamp(row, fields)
        if timestamp is None:
            skipped_rows.append({"unit_id": str(row["id"]), "reason": "missing_timestamp"})
            continue
        local_timestamp = timestamp + offset
        key = (local_timestamp.weekday(), local_timestamp.hour, _daypart(local_timestamp.hour))
        group = groups.setdefault(
            key,
            {
                "weekday": _weekday_name(local_timestamp.weekday()),
                "hour": local_timestamp.hour,
                "daypart": key[2],
                "unit_count": 0,
                "source_projects": set(),
                "content_type_counts": Counter(),
            },
        )
        group["unit_count"] += 1
        group["source_projects"].add(str(row["source_project"]))
        group["content_type_counts"][str(row["content_type"])] += 1

    summary_rows = []
    for key in sorted(groups):
        group = groups[key]
        summary_rows.append(
            {
                "weekday": group["weekday"],
                "hour": group["hour"],
                "daypart": group["daypart"],
                "unit_count": group["unit_count"],
                "source_projects": sorted(group["source_projects"]),
                "content_type_counts": dict(sorted(group["content_type_counts"].items())),
            }
        )

    return {
        "rows": summary_rows,
        "row_count": len(summary_rows),
        "skipped_rows": skipped_rows,
        "skipped_count": len(skipped_rows),
    }


def _connection_from_store(store: Any) -> sqlite3.Connection:
    conn = getattr(store, "conn", store)
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("store must be a Store or sqlite3.Connection")
    conn.row_factory = sqlite3.Row
    return conn


def _validate_unit_timestamp_fields(fields: Sequence[str]) -> tuple[str, ...]:
    if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence):
        raise ValueError("timestamp_fields must be a sequence of non-empty strings")
    normalized = tuple(field for field in fields if isinstance(field, str) and field.strip())
    if len(normalized) != len(fields):
        raise ValueError("timestamp_fields must be a sequence of non-empty strings")
    invalid = sorted(set(normalized) - ALLOWED_TIMESTAMP_FIELDS)
    if invalid:
        raise ValueError("timestamp_fields contains unsupported fields: " + ", ".join(invalid))
    return normalized


def _first_row_timestamp(row: sqlite3.Row, fields: tuple[str, ...]) -> datetime | None:
    for field in fields:
        timestamp = _parse_datetime(row[field])
        if timestamp is not None:
            return timestamp
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


def _daypart(hour: int) -> str:
    if hour < 6:
        return "overnight"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _weekday_name(weekday: int) -> str:
    return ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")[
        weekday
    ]
