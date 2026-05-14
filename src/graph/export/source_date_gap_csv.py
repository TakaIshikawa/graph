"""CSV export for source date gap summaries."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "source_entity_type",
    "dated_unit_count",
    "undated_unit_count",
    "first_date",
    "last_date",
    "largest_gap_days",
    "total_observations",
]
_DATE_FIELDS = ("created_at", "ingested_at", "updated_at")
_METADATA_DATE_KEYS = (
    "observed_at",
    "observed_date",
    "source_date",
    "date",
    "published_at",
    "published_date",
    "created_at",
    "updated_at",
)
_METADATA_DATE_LIST_KEYS = ("observation_dates", "observed_dates", "source_dates", "dates")
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_date_gap_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write source/type date gap statistics as deterministic CSV."""
    unit_list = list(units)
    rows = _gap_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _gap_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"dated_units": 0, "undated_units": 0, "dates": []}
    )
    for unit in units:
        key = (_unit_source(unit), _unit_source_type(unit))
        dates = _unit_dates(unit)
        if dates:
            groups[key]["dated_units"] += 1
            groups[key]["dates"].extend(dates)
        else:
            groups[key]["undated_units"] += 1

    rows: list[dict[str, str | int]] = []
    for source_project, source_entity_type in sorted(
        groups,
        key=lambda key: (_sort_key(key[0]), _sort_key(key[1])),
    ):
        values = sorted(groups[(source_project, source_entity_type)]["dates"])
        first_date = values[0] if values else None
        last_date = values[-1] if values else None
        rows.append(
            {
                "source_project": source_project,
                "source_entity_type": source_entity_type,
                "dated_unit_count": groups[(source_project, source_entity_type)]["dated_units"],
                "undated_unit_count": groups[(source_project, source_entity_type)]["undated_units"],
                "first_date": first_date.isoformat() if first_date else "",
                "last_date": last_date.isoformat() if last_date else "",
                "largest_gap_days": _largest_gap_days(values),
                "total_observations": len(values),
            }
        )
    return rows


def _unit_dates(unit: KnowledgeUnit) -> list[date]:
    values = [_date_value(getattr(unit, field, None)) for field in _DATE_FIELDS]
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    values.extend(_date_value(metadata.get(key)) for key in _METADATA_DATE_KEYS)
    for key in _METADATA_DATE_LIST_KEYS:
        values.extend(_date_value(value) for value in _iter_values(metadata.get(key)))
    return sorted({value for value in values if value is not None})


def _iter_values(value: object) -> list[object]:
    if isinstance(value, list | tuple | set):
        return list(value)
    if value is None:
        return []
    return [value]


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


def _largest_gap_days(values: list[date]) -> int | str:
    if len(values) < 2:
        return ""
    return max((right - left).days for left, right in zip(values, values[1:], strict=False))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_source_type(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_entity_type) or "Unknown"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
