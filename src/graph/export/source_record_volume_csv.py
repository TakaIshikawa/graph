"""CSV export for source record volume by calendar period."""

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

_FIELDNAMES = ["source_project", "source_entity_type", "period", "unit_count", "first_unit_id", "last_unit_id"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_record_volume_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    granularity: str = "month",
    date_metadata_keys: Iterable[str] | None = None,
) -> str | dict[str, Any]:
    """Return or write record volume by month or year."""
    if granularity not in {"month", "year"}:
        raise ValueError("granularity must be 'month' or 'year'")
    unit_list = list(units)
    rows = _volume_rows(unit_list, granularity, tuple(date_metadata_keys or ()))
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "granularity": granularity, "bytes_written": output_path.stat().st_size}


def _volume_rows(units: list[KnowledgeUnit | Mapping[str, Any]], granularity: str, metadata_keys: tuple[str, ...]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str], list[tuple[date, str]]] = defaultdict(list)
    for unit in units:
        unit_date = _unit_date(unit, metadata_keys)
        if unit_date is None:
            continue
        period = unit_date.strftime("%Y-%m" if granularity == "month" else "%Y")
        groups[(_field_value(_get(unit, "source_project")) or "Unknown", _field_value(_get(unit, "source_entity_type")) or "Unknown", period)].append((unit_date, _unit_id(unit)))
    rows: list[dict[str, str | int]] = []
    for (source_project, source_entity_type, period), entries in groups.items():
        ordered = sorted(entries, key=lambda entry: (entry[0], _sort_key(entry[1])))
        rows.append({"source_project": source_project, "source_entity_type": source_entity_type, "period": period, "unit_count": len(entries), "first_unit_id": ordered[0][1], "last_unit_id": ordered[-1][1]})
    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["source_entity_type"]), row["period"]))


def _unit_date(unit: KnowledgeUnit | Mapping[str, Any], metadata_keys: tuple[str, ...]) -> date | None:
    for value in (_get(unit, "created_at"), _get(unit, "updated_at")):
        parsed = _date_value(value)
        if parsed is not None:
            return parsed
    metadata = _metadata(unit)
    for key in metadata_keys:
        parsed = _date_value(metadata.get(key))
        if parsed is not None:
            return parsed
    return None


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


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


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
