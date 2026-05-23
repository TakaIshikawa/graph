"""CSV export for person-like unit metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, normalized_key, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "person_name", "source_field", "normalized_name"]
_PERSON_KEYS = {"author", "authors", "creator", "creators", "contributor", "contributors", "people", "participants", "person"}
_DELIMITER_RE = re.compile(r"\s*(?:;|\||,|\band\b)\s*", re.IGNORECASE)


def export_unit_person_name_inventory_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per normalized person/unit metadata pairing."""
    unit_list = list(units)
    rows = _rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(units: list[Mapping[str, Any] | object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for unit in units:
        uid = unit_id(unit)
        for source_field, name in _person_values(unit):
            normalized = _normalize_name(name)
            key = (uid, source_field, normalized)
            if normalized and key not in seen:
                seen.add(key)
                rows.append({"unit_id": uid, "person_name": name, "source_field": source_field, "normalized_name": normalized})
    return sorted(rows, key=lambda row: (sort_key(row["unit_id"]), sort_key(row["normalized_name"]), sort_key(row["source_field"])))


def _person_values(unit: Mapping[str, Any] | object) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for key in _PERSON_KEYS:
        values.extend((key, name) for name in _split_names(get(unit, key)))
    for key, value in metadata(unit).items():
        source_field = field_value(key)
        if normalized_key(key) in _PERSON_KEYS:
            values.extend((source_field, name) for name in _split_names(value))
    return values


def _split_names(value: object) -> list[str]:
    names: list[str] = []
    for item in flatten_values(value):
        text = field_value(item)
        if not text:
            continue
        names.extend(part for part in _DELIMITER_RE.split(text) if part)
    return [field_value(name) for name in names if field_value(name)]


def _normalize_name(value: object) -> str:
    return re.sub(r"\s+", " ", field_value(value).casefold()).strip()
