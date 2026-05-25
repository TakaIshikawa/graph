"""Store-level source project and entity type mix summaries."""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

DEFAULT_EXAMPLE_LIMIT = 5


def summarize_unit_source_entity_type_mix(
    store: Any,
    *,
    example_limit: int = DEFAULT_EXAMPLE_LIMIT,
) -> dict[str, Any]:
    """Count knowledge units by source project and source entity type."""

    if example_limit < 0:
        raise ValueError("example_limit must be non-negative")
    conn = _connection_from_store(store)
    rows = conn.execute(
        """
        SELECT id, source_project, source_entity_type, content_type, created_at
        FROM knowledge_units
        ORDER BY source_project, source_entity_type, id
        """
    ).fetchall()

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["source_project"]), str(row["source_entity_type"]))
        group = groups.setdefault(
            key,
            {
                "source_project": key[0],
                "source_entity_type": key[1],
                "unit_count": 0,
                "content_type_counts": Counter(),
                "created_at_values": [],
                "example_unit_ids": [],
            },
        )
        group["unit_count"] += 1
        group["content_type_counts"][str(row["content_type"])] += 1
        created_at = _parse_datetime(row["created_at"])
        if created_at is not None:
            group["created_at_values"].append(created_at)
        if len(group["example_unit_ids"]) < example_limit:
            group["example_unit_ids"].append(str(row["id"]))

    summary_rows = []
    for key in sorted(groups):
        group = groups[key]
        created_at_values = group.pop("created_at_values")
        content_type_counts = group.pop("content_type_counts")
        group["content_type_counts"] = dict(sorted(content_type_counts.items()))
        group["earliest_created_at"] = (
            min(created_at_values).isoformat() if created_at_values else None
        )
        group["latest_created_at"] = (
            max(created_at_values).isoformat() if created_at_values else None
        )
        summary_rows.append(group)

    return {"rows": summary_rows, "row_count": len(summary_rows), "unit_count": len(rows)}


def _connection_from_store(store: Any) -> sqlite3.Connection:
    conn = getattr(store, "conn", store)
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("store must be a Store or sqlite3.Connection")
    conn.row_factory = sqlite3.Row
    return conn


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
