"""CSV export for chronological spans by collection metadata."""

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
    "collection",
    "unit_count",
    "source_projects",
    "first_seen_date",
    "last_seen_date",
    "span_days",
    "dated_unit_count",
]
_WHITESPACE_RE = re.compile(r"\s+")
_COLLECTION_KEYS = ("collection", "collection_id", "collection_name", "project", "list", "list_name", "board_name")
_DATE_KEYS = ("date", "source_date", "published_at", "observed_at", "created_at", "updated_at", "ingested_at")


def export_collection_chronology_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write chronology rows for collection-like metadata values."""
    unit_list = list(units)
    rows = _chronology_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "collection_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _chronology_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        for collection in _collections(unit):
            groups[collection].append(unit)

    rows: list[dict[str, str | int]] = []
    for collection, group_units in sorted(groups.items(), key=lambda item: _sort_key(item[0])):
        dated = [_unit_date(unit) for unit in group_units]
        dates = sorted(value for value in dated if value is not None)
        row: dict[str, str | int] = {
            "collection": collection,
            "unit_count": len(group_units),
            "source_projects": _joined_unique(getattr(unit, "source_project", None) for unit in group_units),
            "first_seen_date": "",
            "last_seen_date": "",
            "span_days": "",
            "dated_unit_count": len(dates),
        }
        if dates:
            row["first_seen_date"] = dates[0].isoformat()
            row["last_seen_date"] = dates[-1].isoformat()
            row["span_days"] = (dates[-1] - dates[0]).days
        rows.append(row)
    return rows


def _collections(unit: KnowledgeUnit) -> list[str]:
    metadata = getattr(unit, "metadata", None)
    if not isinstance(metadata, dict):
        return []
    values: dict[str, str] = {}
    for key in _COLLECTION_KEYS:
        if key not in metadata:
            continue
        for value in _iter_values(metadata.get(key)):
            text = _inline_text(value)
            if text:
                values.setdefault(text.casefold(), text)
    return [values[key] for key in sorted(values)]


def _unit_date(unit: KnowledgeUnit) -> date | None:
    metadata = getattr(unit, "metadata", None)
    if isinstance(metadata, dict):
        for key in _DATE_KEYS:
            parsed = _date_value(metadata.get(key))
            if parsed is not None:
                return parsed
    for attr in ("created_at", "updated_at", "ingested_at"):
        parsed = _date_value(getattr(unit, attr, None))
        if parsed is not None:
            return parsed
    return None


def _iter_values(value: object) -> Iterable[object]:
    if isinstance(value, dict):
        return value.values()
    if isinstance(value, list | tuple | set):
        return value
    return [value]


def _joined_unique(values: Iterable[object]) -> str:
    unique = {_field_value(value) for value in values}
    unique.discard("")
    return "; ".join(sorted(unique, key=_sort_key))


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


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
