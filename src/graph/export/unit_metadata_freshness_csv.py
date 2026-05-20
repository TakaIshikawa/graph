"""CSV export for recency of date-like unit metadata values."""

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

_FIELDNAMES = ["source_project", "source_entity_type", "metadata_key", "observed_count", "earliest_value", "latest_value", "days_since_latest"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_metadata_freshness_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    reference_date: date | datetime | str | None = None,
    min_count: int = 1,
) -> str | dict[str, Any]:
    """Return or write metadata date freshness by source/type/key."""
    min_count = _positive_int(min_count, "min_count")
    reference = _date_value(reference_date) if reference_date is not None else date.today()
    if reference is None:
        raise ValueError("reference_date must be a date, datetime, or ISO-like date string")
    unit_list = list(units)
    rows = _freshness_rows(unit_list, reference, min_count)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "min_count": min_count, "bytes_written": output_path.stat().st_size}


def _freshness_rows(units: list[KnowledgeUnit | Mapping[str, Any]], reference: date, min_count: int) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str], list[date]] = defaultdict(list)
    for unit in units:
        source_project = _field_value(_get(unit, "source_project")) or "Unknown"
        source_entity_type = _field_value(_get(unit, "source_entity_type")) or "Unknown"
        for key, value in _metadata(unit).items():
            for parsed in _date_values(value):
                groups[(source_project, source_entity_type, _field_value(key))].append(parsed)
    rows: list[dict[str, str | int]] = []
    for (source_project, source_entity_type, key), values in groups.items():
        if len(values) < min_count:
            continue
        earliest = min(values)
        latest = max(values)
        rows.append({"source_project": source_project, "source_entity_type": source_entity_type, "metadata_key": key, "observed_count": len(values), "earliest_value": earliest.isoformat(), "latest_value": latest.isoformat(), "days_since_latest": (reference - latest).days})
    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["source_entity_type"]), _sort_key(row["metadata_key"])))


def _date_values(value: object) -> list[date]:
    if isinstance(value, Mapping):
        return []
    if isinstance(value, list | tuple | set):
        dates: list[date] = []
        for item in value:
            parsed = _date_value(item)
            if parsed is not None:
                dates.append(parsed)
        return dates
    parsed = _date_value(value)
    return [parsed] if parsed is not None else []


def _date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _field_value(value)
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


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
