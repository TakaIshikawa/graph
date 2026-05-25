"""Store-level duplicate source identifier summaries."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def summarize_unit_duplicate_source_ids(store: Any) -> dict[str, Any]:
    """Find duplicate source IDs within each source project and entity type."""

    conn = _connection_from_store(store)
    rows = conn.execute(
        """
        SELECT id, source_project, source_entity_type, source_id, title, updated_at
        FROM knowledge_units
        ORDER BY source_project, source_entity_type, source_id, id
        """
    ).fetchall()

    groups: dict[tuple[str, str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["source_project"]),
                str(row["source_entity_type"]),
                str(row["source_id"]),
            )
        ].append(row)

    duplicate_rows = []
    for key in sorted(groups):
        grouped_rows = groups[key]
        if len(grouped_rows) <= 1:
            continue
        ordered_rows = sorted(grouped_rows, key=lambda row: str(row["id"]))
        updated_at_values = [
            parsed for parsed in (_parse_datetime(row["updated_at"]) for row in ordered_rows) if parsed
        ]
        duplicate_rows.append(
            {
                "source_project": key[0],
                "source_entity_type": key[1],
                "source_id": key[2],
                "unit_count": len(ordered_rows),
                "unit_ids": [str(row["id"]) for row in ordered_rows],
                "titles": [str(row["title"]) for row in ordered_rows],
                "latest_updated_at": (
                    max(updated_at_values).isoformat() if updated_at_values else None
                ),
            }
        )

    return {
        "duplicate_count": len(duplicate_rows),
        "duplicate_groups": duplicate_rows,
        "rows": duplicate_rows,
    }


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
