"""CSV export helpers for tag timeline reports."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["period", "tag", "count", "source_project", "source_entity_type"]
_GRANULARITIES = {"day", "week", "month", "year"}


def export_tag_timeline_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    granularity: str = "month",
    date_metadata_keys: Iterable[str] | None = None,
) -> str | dict[str, Any]:
    """Return or write tag counts over deterministic time periods."""
    if granularity not in _GRANULARITIES:
        raise ValueError("granularity must be one of: day, week, month, year")

    rows = _timeline_rows(list(units), granularity, list(date_metadata_keys or []))
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "rows_written": len(rows),
        "granularity": granularity,
        "bytes_written": output_path.stat().st_size,
    }


def _timeline_rows(
    units: list[KnowledgeUnit],
    granularity: str,
    date_metadata_keys: list[str],
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for unit in units:
        timestamp = _unit_datetime(unit, date_metadata_keys)
        if timestamp is None:
            continue
        period = _period_label(timestamp.date(), granularity)
        source_project = _field_value(unit.source_project)
        source_entity_type = _text(unit.source_entity_type)
        for tag in _unit_tags(unit):
            counts[(period, tag, source_project, source_entity_type)] += 1

    return [
        {
            "period": period,
            "tag": tag,
            "count": count,
            "source_project": source_project,
            "source_entity_type": source_entity_type,
        }
        for (period, tag, source_project, source_entity_type), count in sorted(
            counts.items(),
            key=lambda item: (
                item[0][0],
                _sort_key(item[0][1]),
                _sort_key(item[0][2]),
                _sort_key(item[0][3]),
            ),
        )
    ]


def _period_label(value: date, granularity: str) -> str:
    if granularity == "day":
        return value.isoformat()
    if granularity == "week":
        return (value - timedelta(days=value.weekday())).isoformat()
    if granularity == "year":
        return f"{value.year:04d}"
    return f"{value.year:04d}-{value.month:02d}"


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted({_inline_text(tag) for tag in unit.tags if _inline_text(tag)}, key=_sort_key)


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
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _text(value: object) -> str:
    return str(value or "")
