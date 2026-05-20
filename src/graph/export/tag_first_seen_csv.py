"""CSV export for earliest observed unit per tag."""

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

_FIELDNAMES = ["tag", "first_seen", "unit_id", "unit_title", "source", "total_units_with_tag"]
_DATE_KEYS = ("observed_at", "observed_date", "source_date", "published_at", "published_date", "created_at", "date")
_WHITESPACE_RE = re.compile(r"\s+")


def export_tag_first_seen_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write the first dated unit observed for each tag."""
    unit_list = list(units)
    rows = _first_seen_rows(unit_list)
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


def _first_seen_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    entries: dict[str, list[tuple[date | None, KnowledgeUnit | Mapping[str, Any]]]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    for unit in units:
        unit_id = _unit_id(unit)
        for tag in _unit_tags(unit):
            pair = (tag, unit_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            entries[tag].append((_unit_date(unit), unit))

    rows: list[dict[str, str | int]] = []
    for tag, tag_entries in entries.items():
        first_seen, first_unit = sorted(tag_entries, key=_entry_sort_key)[0]
        rows.append(
            {
                "tag": tag,
                "first_seen": first_seen.isoformat() if first_seen is not None else "",
                "unit_id": _unit_id(first_unit),
                "unit_title": _field_value(_get(first_unit, "title")),
                "source": _field_value(_get(first_unit, "source_project")) or "Unknown",
                "total_units_with_tag": len(tag_entries),
            }
        )
    return sorted(rows, key=lambda row: (row["first_seen"] == "", row["first_seen"], _sort_key(row["tag"])))


def _entry_sort_key(entry: tuple[date | None, KnowledgeUnit | Mapping[str, Any]]) -> tuple[int, date, tuple[str, str], tuple[str, str]]:
    seen_date, unit = entry
    return (
        1 if seen_date is None else 0,
        seen_date or date.max,
        _sort_key(_unit_id(unit)),
        _sort_key(_field_value(_get(unit, "title"))),
    )


def _unit_date(unit: KnowledgeUnit | Mapping[str, Any]) -> date | None:
    metadata = _metadata(unit)
    for key in _DATE_KEYS:
        parsed = _date_value(metadata.get(key))
        if parsed is not None:
            return parsed
    for key in (*_DATE_KEYS, "updated_at", "ingested_at"):
        parsed = _date_value(_get(unit, key))
        if parsed is not None:
            return parsed
    return None


def _unit_tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    tags = _get(unit, "tags", [])
    if isinstance(tags, str):
        return [_field_value(tags)] if _field_value(tags) else []
    if isinstance(tags, Iterable):
        return sorted({_field_value(tag) for tag in tags if _field_value(tag)}, key=_sort_key)
    return []


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


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
