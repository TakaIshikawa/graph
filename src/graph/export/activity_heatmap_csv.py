"""CSV export helpers for activity heatmap reports."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["date", "hour", "count", "source_project", "source_entity_type"]


def export_units_to_activity_heatmap_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    date_metadata_keys: Iterable[str] | None = None,
) -> str | dict[str, Any]:
    """Return or write unit activity counts by date, hour, source, and entity type."""
    rows = _heatmap_rows(list(units), list(date_metadata_keys or []))
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "rows_written": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _heatmap_rows(
    units: list[KnowledgeUnit],
    date_metadata_keys: list[str],
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, int, str, str]] = Counter()
    for unit in units:
        timestamp = _unit_datetime(unit, date_metadata_keys)
        if timestamp is None:
            continue
        counts[
            (
                timestamp.date().isoformat(),
                timestamp.hour,
                _field_value(unit.source_project),
                _text(unit.source_entity_type),
            )
        ] += 1

    return [
        {
            "date": day,
            "hour": hour,
            "count": count,
            "source_project": source_project,
            "source_entity_type": source_entity_type,
        }
        for (day, hour, source_project, source_entity_type), count in sorted(counts.items())
    ]


def _unit_datetime(unit: KnowledgeUnit, date_metadata_keys: list[str]) -> datetime | None:
    for key in date_metadata_keys:
        parsed = _parse_datetime(_metadata_value(unit.metadata, key))
        if parsed is not None:
            return parsed
    for value in (unit.created_at, unit.updated_at, unit.ingested_at):
        parsed = _parse_datetime(value)
        if parsed is not None:
            return parsed
    return None


def _metadata_value(metadata: dict, key: str) -> Any:
    value: Any = metadata
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed_date = date.fromisoformat(text)
        except ValueError:
            return None
        return datetime(parsed_date.year, parsed_date.month, parsed_date.day)
    return parsed


def _render_csv(rows: list[dict[str, Any]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _text(getattr(value, "value", value))


def _text(value: object) -> str:
    return str(value or "")
