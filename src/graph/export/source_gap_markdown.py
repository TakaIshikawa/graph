"""Markdown export for large timestamp gaps within source/entity groups."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_DATE_FIELDS = {"created_at", "ingested_at", "updated_at"}
_SECONDS_PER_DAY = 24 * 60 * 60
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_gap_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    date_field: str = "created_at",
    gap_days: int = 30,
    max_gaps: int = 10,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown report of large gaps between source timestamps."""
    if date_field not in _DATE_FIELDS:
        valid = ", ".join(sorted(_DATE_FIELDS))
        raise ValueError(f"date_field must be one of: {valid}")
    if not isinstance(gap_days, int) or isinstance(gap_days, bool) or gap_days < 1:
        raise ValueError("gap_days must be a positive integer")
    if not isinstance(max_gaps, int) or isinstance(max_gaps, bool) or max_gaps < 1:
        raise ValueError("max_gaps must be a positive integer")

    dated_units = sorted(
        [
            (_normalize_datetime(getattr(unit, date_field, None)), unit)
            for unit in units
            if _normalize_datetime(getattr(unit, date_field, None)) is not None
        ],
        key=lambda item: (_group_key(item[1]), _datetime_key(item[0]), _unit_sort_key(item[1])),
    )
    rows = _gap_rows(dated_units, gap_days=gap_days, max_gaps=max_gaps)
    text = _render_report(
        rows,
        units_scanned=len(dated_units),
        date_field=date_field,
        gap_days=gap_days,
        max_gaps=max_gaps,
    )

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "units_scanned": len(dated_units),
        "gaps_exported": len(rows),
        "date_field": date_field,
        "gap_days": gap_days,
        "max_gaps": max_gaps,
        "bytes_written": output_path.stat().st_size,
    }


def _gap_rows(
    dated_units: list[tuple[datetime, KnowledgeUnit]],
    *,
    gap_days: int,
    max_gaps: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[datetime, KnowledgeUnit]]] = defaultdict(list)
    for timestamp, unit in dated_units:
        grouped[_group_key(unit)].append((timestamp, unit))

    rows = []
    for (source_project, source_entity_type), group_items in grouped.items():
        sorted_items = sorted(group_items, key=lambda item: (_datetime_key(item[0]), _unit_sort_key(item[1])))
        for (previous_timestamp, _previous_unit), (next_timestamp, _next_unit) in zip(sorted_items, sorted_items[1:]):
            gap_length = (next_timestamp - previous_timestamp).total_seconds() / _SECONDS_PER_DAY
            if gap_length <= gap_days:
                continue
            rows.append(
                {
                    "source_project": source_project,
                    "source_entity_type": source_entity_type,
                    "previous_timestamp": previous_timestamp,
                    "next_timestamp": next_timestamp,
                    "gap_days": gap_length,
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            -row["gap_days"],
            _sort_key(row["source_project"]),
            _sort_key(row["source_entity_type"]),
            _datetime_key(row["previous_timestamp"]),
            _datetime_key(row["next_timestamp"]),
        ),
    )[:max_gaps]


def _render_report(
    rows: list[dict[str, Any]],
    *,
    units_scanned: int,
    date_field: str,
    gap_days: int,
    max_gaps: int,
) -> str:
    lines = [
        "# Source Gaps",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Units scanned | {units_scanned} |",
        f"| Gaps reported | {len(rows)} |",
        f"| Date field | {_markdown_cell(date_field)} |",
        f"| Gap days | {gap_days} |",
        f"| Max gaps | {max_gaps} |",
        "",
        "## Gaps",
        "",
        "| Source project | Entity type | Previous timestamp | Next timestamp | Gap days |",
        "| --- | --- | --- | --- | ---: |",
    ]
    if rows:
        for row in rows:
            lines.append(
                "| "
                f"{_markdown_cell(row['source_project'])} | "
                f"{_markdown_cell(row['source_entity_type'])} | "
                f"{_markdown_cell(_datetime_text(row['previous_timestamp']))} | "
                f"{_markdown_cell(_datetime_text(row['next_timestamp']))} | "
                f"{_gap_days_text(row['gap_days'])} |"
            )
    else:
        lines.append("| _None_ | _None_ | _None_ | _None_ | 0.0 |")
    return "\n".join(lines).rstrip() + "\n"


def _group_key(unit: KnowledgeUnit) -> tuple[str, str]:
    return (_unit_source(unit), _entity_type(unit))


def _unit_source(unit: KnowledgeUnit) -> str:
    return _inline_text(getattr(unit.source_project, "value", unit.source_project)) or "Unknown"


def _entity_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or "Unknown"


def _normalize_datetime(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _datetime_key(value: datetime | None) -> str:
    return value.isoformat() if isinstance(value, datetime) else ""


def _datetime_text(value: datetime) -> str:
    return value.isoformat()


def _gap_days_text(value: float) -> str:
    return f"{value:.1f}"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str, str]:
    return (_unit_source(unit), _inline_text(unit.source_id), _inline_text(unit.title), _inline_text(unit.id))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
