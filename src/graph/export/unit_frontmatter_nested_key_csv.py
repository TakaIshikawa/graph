"""CSV export for nested unit frontmatter and metadata keys."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "key_path", "value_type", "depth"]


def export_unit_frontmatter_nested_key_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["key_path"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for root, value in (("metadata", metadata(unit)), ("frontmatter", get(unit, "frontmatter"))):
        if isinstance(value, Mapping):
            for path, child in _flatten(value, root):
                rows.append({"unit_id": uid, "title": title, "key_path": path, "value_type": _value_type(child), "depth": path.count(".") + 1})
    return rows


def _flatten(value: Mapping[str, Any], prefix: str) -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    for key, child in value.items():
        path = f"{prefix}.{field_value(key)}"
        rows.append((path, child))
        if isinstance(child, Mapping):
            rows.extend(_flatten(child, path))
    return rows


def _value_type(value: object) -> str:
    if isinstance(value, Mapping):
        return "dict"
    if isinstance(value, list | tuple | set):
        return "list"
    if value is None:
        return "null"
    return type(value).__name__
