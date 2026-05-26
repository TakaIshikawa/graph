"""CSV export for unit identifier quality signals."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = [
    "unit_id",
    "source",
    "entity_type",
    "identifier_count",
    "identifier_types",
    "missing_canonical_id",
    "duplicate_identifier_values",
    "quality_flags",
]
_IDENTIFIER_KEYS = ("doi", "isbn", "url", "source_id", "external_ids")
_CANONICAL_KEYS = ("canonical_id", "id", "unit_id", "doi", "isbn", "url", "source_id")


def export_units_to_identifier_quality_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str]:
    identifiers = _identifiers(unit)
    values = [value.casefold() for _, value in identifiers]
    duplicates = sorted({value for value, count in Counter(values).items() if count > 1}, key=sort_key)
    missing_canonical = not any(field_value(get(unit, key) or metadata(unit).get(key)) for key in _CANONICAL_KEYS)
    flags: list[str] = []
    if not identifiers:
        flags.append("no_identifiers")
    if missing_canonical:
        flags.append("missing_canonical_id")
    if duplicates:
        flags.append("duplicate_identifier_values")
    return {
        "unit_id": unit_id(unit),
        "source": field_value(get(unit, "source_project") or metadata(unit).get("source") or metadata(unit).get("source_project")),
        "entity_type": field_value(get(unit, "source_entity_type") or get(unit, "entity_type") or metadata(unit).get("entity_type")),
        "identifier_count": str(len(identifiers)),
        "identifier_types": "; ".join(sorted({kind for kind, _ in identifiers}, key=sort_key)),
        "missing_canonical_id": str(missing_canonical).lower(),
        "duplicate_identifier_values": "; ".join(duplicates),
        "quality_flags": "; ".join(flags),
    }


def _identifiers(unit: Mapping[str, Any] | object) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    meta = metadata(unit)
    for key in _IDENTIFIER_KEYS:
        raw = get(unit, key)
        if raw in (None, "", []):
            raw = meta.get(key)
        if key == "external_ids" and isinstance(raw, Mapping):
            for child_key, child_value in raw.items():
                for value in flatten_values(child_value):
                    text = field_value(value)
                    if text:
                        found.append((field_value(child_key) or "external_id", text))
            continue
        for value in flatten_values(raw):
            text = field_value(value)
            if text:
                found.append((key, text))
    return found
