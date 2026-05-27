"""CSV export for boolean-like unit metadata values."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "metadata_key", "raw_value", "normalized_value"]
_TRUE = {"true", "yes", "done", "complete", "completed", "open"}
_FALSE = {"false", "no", "closed", "cancelled", "canceled"}


def export_unit_metadata_boolean_flags_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        for key, value in _walk(metadata(unit)):
            normalized = _normalized(value)
            if normalized:
                rows.append({"unit_id": unit_id(unit), "title": title, "metadata_key": key, "raw_value": field_value(value), "normalized_value": normalized})
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["metadata_key"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _walk(value: Any, prefix: str = "metadata") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        return [item for key in sorted(value, key=sort_key) for item in _walk(value[key], f"{prefix}.{field_value(key)}")]
    if isinstance(value, list | tuple):
        return [item for index, child in enumerate(value) for item in _walk(child, f"{prefix}.{index}")]
    return [(prefix, value)]


def _normalized(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = field_value(value).casefold()
    if text in _TRUE:
        return "true"
    if text in _FALSE:
        return "false"
    if text in {"unknown", "n/a", "na"}:
        return "unknown"
    return ""
