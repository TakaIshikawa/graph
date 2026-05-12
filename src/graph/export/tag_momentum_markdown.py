"""Markdown export for tag momentum across adjacent time windows."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_DATE_FIELDS = {"created_at", "ingested_at", "updated_at"}
_WHITESPACE_RE = re.compile(r"\s+")


def export_tag_momentum_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    recent_days: int = 30,
    previous_days: int = 30,
    date_field: str = "updated_at",
    min_recent_count: int = 1,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown report comparing recent tag counts."""
    if not isinstance(recent_days, int) or isinstance(recent_days, bool) or recent_days < 1:
        raise ValueError("recent_days must be a positive integer")
    if not isinstance(previous_days, int) or isinstance(previous_days, bool) or previous_days < 1:
        raise ValueError("previous_days must be a positive integer")
    if date_field not in _DATE_FIELDS:
        valid = ", ".join(sorted(_DATE_FIELDS))
        raise ValueError(f"date_field must be one of: {valid}")
    if (
        not isinstance(min_recent_count, int)
        or isinstance(min_recent_count, bool)
        or min_recent_count < 1
    ):
        raise ValueError("min_recent_count must be a positive integer")

    dated_units = sorted(
        [
            (_normalize_datetime(getattr(unit, date_field, None)), unit)
            for unit in units
            if _normalize_datetime(getattr(unit, date_field, None)) is not None
        ],
        key=lambda item: (_datetime_key(item[0]), _unit_sort_key(item[1])),
    )
    anchor = max((timestamp for timestamp, _unit in dated_units), default=None)
    rows: list[dict[str, Any]] = []
    recent_count = 0
    previous_count = 0
    if anchor is not None:
        recent_start = anchor - timedelta(days=recent_days)
        previous_start = recent_start - timedelta(days=previous_days)
        recent_tags: Counter[str] = Counter()
        previous_tags: Counter[str] = Counter()
        for timestamp, unit in dated_units:
            if recent_start < timestamp <= anchor:
                recent_count += 1
                recent_tags.update(_unit_tags(unit))
            elif previous_start < timestamp <= recent_start:
                previous_count += 1
                previous_tags.update(_unit_tags(unit))
        rows = _momentum_rows(recent_tags, previous_tags, min_recent_count=min_recent_count)

    text = _render_report(
        rows,
        units_scanned=len(dated_units),
        recent_unit_count=recent_count,
        previous_unit_count=previous_count,
        newest_timestamp=anchor,
        recent_days=recent_days,
        previous_days=previous_days,
        date_field=date_field,
        min_recent_count=min_recent_count,
    )

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "rows_exported": len(rows),
        "units_scanned": len(dated_units),
        "recent_unit_count": recent_count,
        "previous_unit_count": previous_count,
        "recent_days": recent_days,
        "previous_days": previous_days,
        "date_field": date_field,
        "min_recent_count": min_recent_count,
        "bytes_written": output_path.stat().st_size,
    }


def _momentum_rows(
    recent_tags: Counter[str],
    previous_tags: Counter[str],
    *,
    min_recent_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for tag, recent_count in recent_tags.items():
        if recent_count < min_recent_count:
            continue
        previous_count = previous_tags[tag]
        delta = recent_count - previous_count
        rows.append(
            {
                "tag": tag,
                "recent_count": recent_count,
                "previous_count": previous_count,
                "delta": delta,
                "percent_change": None if previous_count == 0 else (delta / previous_count) * 100,
            }
        )
    return sorted(
        rows,
        key=lambda row: (-row["recent_count"], -row["delta"], _sort_key(row["tag"])),
    )


def _render_report(
    rows: list[dict[str, Any]],
    *,
    units_scanned: int,
    recent_unit_count: int,
    previous_unit_count: int,
    newest_timestamp: datetime | None,
    recent_days: int,
    previous_days: int,
    date_field: str,
    min_recent_count: int,
) -> str:
    lines = [
        "# Tag Momentum",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Units scanned | {units_scanned} |",
        f"| Recent units | {recent_unit_count} |",
        f"| Previous units | {previous_unit_count} |",
        f"| Tags reported | {len(rows)} |",
        f"| Date field | {_markdown_cell(date_field)} |",
        f"| Newest timestamp | {_markdown_cell(_datetime_text(newest_timestamp))} |",
        f"| Recent days | {recent_days} |",
        f"| Previous days | {previous_days} |",
        f"| Min recent count | {min_recent_count} |",
        "",
        "## Tags",
        "",
        "| Tag | Recent | Previous | Delta | Percent change |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if rows:
        for row in rows:
            lines.append(
                "| "
                f"{_markdown_cell(row['tag'])} | "
                f"{row['recent_count']} | "
                f"{row['previous_count']} | "
                f"{_signed_int(row['delta'])} | "
                f"{_percent_text(row['percent_change'])} |"
            )
    else:
        lines.append("| _None_ | 0 | 0 | +0 | _N/A_ |")
    return "\n".join(lines).rstrip() + "\n"


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted({_inline_text(tag) for tag in unit.tags if _inline_text(tag)}, key=_sort_key)


def _normalize_datetime(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _datetime_key(value: datetime | None) -> str:
    return value.isoformat() if isinstance(value, datetime) else ""


def _datetime_text(value: datetime | None) -> str:
    return value.isoformat() if isinstance(value, datetime) else "_None_"


def _signed_int(value: int) -> str:
    return f"{value:+d}"


def _percent_text(value: float | None) -> str:
    if value is None:
        return "_N/A_"
    return f"{value:+.1f}%"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str, str]:
    return (_unit_source(unit), _inline_text(unit.source_id), _inline_text(unit.title), _inline_text(unit.id))


def _unit_source(unit: KnowledgeUnit) -> str:
    return _inline_text(getattr(unit.source_project, "value", unit.source_project)) or "Unknown"


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
