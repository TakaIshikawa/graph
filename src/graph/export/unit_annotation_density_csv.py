"""CSV export for unit annotation density."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = [
    "unit_id",
    "title",
    "source",
    "content_length",
    "annotation_count",
    "highlight_count",
    "note_count",
    "density_per_1k_chars",
    "density_bucket",
]
_ANNOTATION_KEYS = ("annotations", "annotation_count", "comments", "comment_count")
_HIGHLIGHT_KEYS = ("highlights", "highlight_count")
_NOTE_KEYS = ("notes", "note_count", "margin_notes", "margin_note_count")


def export_units_to_annotation_density_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write per-unit annotation density rows."""
    unit_list = list(units)
    rows = _rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(units: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    rows = []
    for unit in units:
        content_length = len(field_value(get(unit, "content") or get(unit, "text") or get(unit, "body")))
        annotation_count = _count_metadata_values(unit, _ANNOTATION_KEYS)
        highlight_count = _count_metadata_values(unit, _HIGHLIGHT_KEYS)
        note_count = _count_metadata_values(unit, _NOTE_KEYS)
        total = annotation_count + highlight_count + note_count
        density = 0.0 if content_length <= 0 else total * 1000 / content_length
        rows.append(
            {
                "unit_id": unit_id(unit),
                "title": field_value(get(unit, "title")),
                "source": field_value(get(unit, "source_project") or get(unit, "source")),
                "content_length": content_length,
                "annotation_count": annotation_count,
                "highlight_count": highlight_count,
                "note_count": note_count,
                "density_per_1k_chars": f"{density:.2f}",
                "density_bucket": _bucket(density, total),
            }
        )
    return sorted(rows, key=lambda row: (-float(str(row["density_per_1k_chars"])), sort_key(row["unit_id"])))


def _count_metadata_values(unit: Mapping[str, Any] | object, keys: tuple[str, ...]) -> int:
    data = metadata(unit)
    total = 0
    for key in keys:
        total += _count_value(data.get(key))
    return total


def _count_value(value: object) -> int:
    if value is None or value is False:
        return 0
    if isinstance(value, bool):
        return 1
    if isinstance(value, int | float):
        return max(0, int(value))
    if isinstance(value, Mapping):
        return len(value) if value else 0
    flattened = flatten_values(value)
    return len(flattened) if flattened else (1 if field_value(value) else 0)


def _bucket(density: float, total: int) -> str:
    if total <= 0:
        return "none"
    if density < 5:
        return "low"
    if density < 20:
        return "medium"
    return "high"
