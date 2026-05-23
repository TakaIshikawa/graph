"""Markdown export for collection reading progress."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import (
    field_value,
    flatten_values,
    get,
    metadata,
    normalized_key,
    sort_key,
    write_csv,
)

_COLLECTION_KEYS = {
    "collection",
    "collections",
    "collection_id",
    "collection_name",
    "project",
    "list",
    "folder",
}
_UNASSIGNED = "unassigned"


def export_collection_reading_progress_markdown(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write a Markdown table summarizing reading progress by collection."""
    unit_list = list(units)
    rows = _rows(unit_list)
    text = _render(rows)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {
        "path": output_path,
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": bytes_written,
    }


def _rows(units: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for unit in units:
        progress = _progress(unit)
        status = _status(unit, progress)
        for collection in _collections(unit):
            groups[collection].append((status, progress))
    rows = []
    for collection, values in groups.items():
        rows.append(
            {
                "collection": collection,
                "total_units": len(values),
                "not_started": sum(1 for status, _ in values if status == "not_started"),
                "in_progress": sum(1 for status, _ in values if status == "in_progress"),
                "completed": sum(1 for status, _ in values if status == "completed"),
                "average_progress_percent": f"{sum(progress for _, progress in values) / len(values):.1f}"
                if values
                else "0.0",
            }
        )
    return sorted(rows, key=lambda row: sort_key(row["collection"]))


def _collections(unit: Mapping[str, Any] | object) -> list[str]:
    values: set[str] = set()
    for key in _COLLECTION_KEYS:
        text = field_value(get(unit, key))
        if text:
            values.add(text)
    for key, value in metadata(unit).items():
        if normalized_key(key) in _COLLECTION_KEYS:
            values.update(field_value(item) for item in flatten_values(value) if field_value(item))
    return sorted(values, key=sort_key) or [_UNASSIGNED]


def _progress(unit: Mapping[str, Any] | object) -> float:
    data = metadata(unit)
    explicit = _number(data.get("progress") or data.get("progress_percent"))
    if explicit is not None:
        return _clamp(explicit * 100 if 0 <= explicit <= 1 else explicit)
    pages_read = _number(data.get("pages_read") or data.get("current_page"))
    total_pages = _number(data.get("total_pages") or data.get("pages"))
    if pages_read is not None and total_pages and total_pages > 0:
        return _clamp((pages_read / total_pages) * 100)
    status = field_value(data.get("status") or data.get("reading_status")).casefold()
    return 100.0 if status in {"completed", "complete", "done", "finished", "read"} else 0.0


def _status(unit: Mapping[str, Any] | object, progress: float) -> str:
    raw = field_value(
        metadata(unit).get("status") or metadata(unit).get("reading_status")
    ).casefold()
    if raw in {"completed", "complete", "done", "finished", "read"} or progress >= 100:
        return "completed"
    if raw in {"in_progress", "reading", "started"} or progress > 0:
        return "in_progress"
    return "not_started"


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        text = field_value(value).removesuffix("%")
        return float(text) if text else None
    except ValueError:
        return None


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _render(rows: list[dict[str, str | int]]) -> str:
    lines = [
        "| collection | total_units | not_started | in_progress | completed | average_progress_percent |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['collection']} | {row['total_units']} | {row['not_started']} | {row['in_progress']} | {row['completed']} | {row['average_progress_percent']} |"
        for row in rows
    )
    return "\n".join(lines) + "\n"
