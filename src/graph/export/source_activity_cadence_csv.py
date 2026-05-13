"""CSV export for source activity cadence by source and entity type."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "source_entity_type",
    "observation_count",
    "first_seen_date",
    "last_seen_date",
    "active_span_days",
    "average_gap_days",
    "max_gap_days",
]
_WHITESPACE_RE = re.compile(r"\s+")
_DATE_METADATA_KEYS = (
    "date",
    "source_date",
    "published_at",
    "observed_at",
    "created_at",
    "updated_at",
    "ingested_at",
)


def export_source_activity_cadence_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write source/type cadence rows calculated from unit dates."""
    unit_list = list(units)
    rows = _cadence_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "group_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _cadence_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[date | None]] = defaultdict(list)
    for unit in units:
        key = (
            _field_value(getattr(unit, "source_project", None)) or "Unknown",
            _field_value(getattr(unit, "source_entity_type", None)) or "Unknown",
        )
        groups[key].append(_unit_date(unit))

    rows: list[dict[str, str | int]] = []
    for (source_project, source_entity_type), values in sorted(
        groups.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))
    ):
        dates = sorted(value for value in values if value is not None)
        row: dict[str, str | int] = {
            "source_project": source_project,
            "source_entity_type": source_entity_type,
            "observation_count": len(values),
            "first_seen_date": "",
            "last_seen_date": "",
            "active_span_days": "",
            "average_gap_days": "",
            "max_gap_days": "",
        }
        if dates:
            row["first_seen_date"] = dates[0].isoformat()
            row["last_seen_date"] = dates[-1].isoformat()
            row["active_span_days"] = (dates[-1] - dates[0]).days
            gaps = [(right - left).days for left, right in zip(dates, dates[1:])]
            if gaps:
                row["average_gap_days"] = _format_number(sum(gaps) / len(gaps))
                row["max_gap_days"] = max(gaps)
        rows.append(row)
    return rows


def _unit_date(unit: KnowledgeUnit) -> date | None:
    metadata = getattr(unit, "metadata", None)
    if isinstance(metadata, dict):
        for key in _DATE_METADATA_KEYS:
            parsed = _date_value(metadata.get(key))
            if parsed is not None:
                return parsed
    for attr in ("created_at", "updated_at", "ingested_at"):
        parsed = _date_value(getattr(unit, attr, None))
        if parsed is not None:
            return parsed
    return None


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _inline_text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _format_number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
