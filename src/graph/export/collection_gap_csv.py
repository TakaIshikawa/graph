"""CSV export for gaps in collection-like unit sequences."""

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

_FIELDNAMES = ["collection", "source_project", "gap_type", "previous_unit_id", "next_unit_id", "gap_size", "detail"]
_COLLECTION_KEYS = ("collection", "collection_id", "collection_name", "project", "list", "list_name", "shelf", "playlist")
_SEQUENCE_KEYS = ("index", "position", "order", "sequence")
_DATE_KEYS = ("date", "source_date", "published_at", "observed_at", "created_at", "updated_at", "ingested_at")
_WHITESPACE_RE = re.compile(r"\s+")
_LARGE_DATE_GAP_DAYS = 30


def export_collection_gap_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write missing sequence and large date gaps within collections."""
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
        "gap_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _gap_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[KnowledgeUnit | Mapping[str, Any]]] = defaultdict(list)
    for unit in units:
        source_project = _field_value(_get(unit, "source_project")) or "Unknown"
        for collection in _collections(unit):
            groups[(collection, source_project)].append(unit)

    rows: list[dict[str, str | int]] = []
    for (collection, source_project), group_units in sorted(
        groups.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))
    ):
        rows.extend(_sequence_gap_rows(collection, source_project, group_units))
        rows.extend(_date_gap_rows(collection, source_project, group_units))
    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["collection"]),
            _sort_key(row["source_project"]),
            _sort_key(row["gap_type"]),
            _sort_key(row["previous_unit_id"]),
            _sort_key(row["next_unit_id"]),
        ),
    )


def _sequence_gap_rows(
    collection: str,
    source_project: str,
    units: list[KnowledgeUnit | Mapping[str, Any]],
) -> list[dict[str, str | int]]:
    sequenced = sorted(
        ((sequence, unit) for unit in units if (sequence := _sequence(unit)) is not None),
        key=lambda item: (item[0], _sort_key(_unit_id(item[1]))),
    )
    rows: list[dict[str, str | int]] = []
    for (previous_sequence, previous_unit), (next_sequence, next_unit) in zip(sequenced, sequenced[1:]):
        if next_sequence - previous_sequence <= 1:
            continue
        gap_size = next_sequence - previous_sequence - 1
        rows.append(
            {
                "collection": collection,
                "source_project": source_project,
                "gap_type": "missing_sequence",
                "previous_unit_id": _unit_id(previous_unit),
                "next_unit_id": _unit_id(next_unit),
                "gap_size": gap_size,
                "detail": f"missing sequence values after {previous_sequence} before {next_sequence}",
            }
        )
    return rows


def _date_gap_rows(
    collection: str,
    source_project: str,
    units: list[KnowledgeUnit | Mapping[str, Any]],
) -> list[dict[str, str | int]]:
    dated = sorted(
        ((unit_date, unit) for unit in units if (unit_date := _unit_date(unit)) is not None),
        key=lambda item: (item[0], _sort_key(_unit_id(item[1]))),
    )
    rows: list[dict[str, str | int]] = []
    for (previous_date, previous_unit), (next_date, next_unit) in zip(dated, dated[1:]):
        gap_size = (next_date - previous_date).days
        if gap_size <= _LARGE_DATE_GAP_DAYS:
            continue
        rows.append(
            {
                "collection": collection,
                "source_project": source_project,
                "gap_type": "large_date_gap",
                "previous_unit_id": _unit_id(previous_unit),
                "next_unit_id": _unit_id(next_unit),
                "gap_size": gap_size,
                "detail": f"{previous_date.isoformat()} to {next_date.isoformat()}",
            }
        )
    return rows


def _collections(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    metadata = _metadata(unit)
    values: dict[str, str] = {}
    for key in _COLLECTION_KEYS:
        if key not in metadata:
            continue
        for value in _iter_values(metadata.get(key)):
            text = _inline_text(value)
            if text:
                values.setdefault(text.casefold(), text)
    return [values[key] for key in sorted(values)]


def _sequence(unit: KnowledgeUnit | Mapping[str, Any]) -> int | None:
    metadata = _metadata(unit)
    for key in _SEQUENCE_KEYS:
        value = metadata.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _unit_date(unit: KnowledgeUnit | Mapping[str, Any]) -> date | None:
    metadata = _metadata(unit)
    for key in _DATE_KEYS:
        parsed = _date_value(metadata.get(key))
        if parsed is not None:
            return parsed
    for attr in ("created_at", "updated_at", "ingested_at"):
        parsed = _date_value(_get(unit, attr))
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


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _iter_values(value: object) -> Iterable[object]:
    if isinstance(value, Mapping):
        return value.values()
    if isinstance(value, list | tuple | set):
        return value
    return [value]


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
