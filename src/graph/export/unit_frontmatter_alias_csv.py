"""CSV export for unit frontmatter alias values."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, normalized_key, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "alias", "index", "source_field"]
_ALIAS_KEYS = {"alias", "aliases"}


def export_unit_frontmatter_alias_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["source_field"]), int(row["index"]), sort_key(row["alias"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for source_field, value in _alias_fields(unit):
        values = value if isinstance(value, list | tuple) else [value]
        for index, alias in enumerate(values):
            text = field_value(alias)
            if text:
                rows.append({"unit_id": uid, "title": title, "alias": text, "index": index, "source_field": source_field})
    return rows


def _alias_fields(unit: Mapping[str, Any] | object) -> list[tuple[str, object]]:
    fields: list[tuple[str, object]] = []
    meta = metadata(unit)
    for key, value in meta.items():
        if normalized_key(key) in _ALIAS_KEYS:
            fields.append((f"metadata.{field_value(key)}", value))
    frontmatter = get(unit, "frontmatter")
    if isinstance(frontmatter, Mapping):
        for key, value in frontmatter.items():
            if normalized_key(key) in _ALIAS_KEYS:
                fields.append((f"frontmatter.{field_value(key)}", value))
    return fields
