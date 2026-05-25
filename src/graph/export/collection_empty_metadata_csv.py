"""CSV export for collection-level empty metadata counts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, write_csv

_FIELDNAMES = ["collection", "total_units", "missing_title", "missing_tags", "missing_source", "missing_timestamps", "blank_content", "missing_rate"]


def export_collection_empty_metadata_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {key: 0 for key in _FIELDNAMES if key not in {"collection", "missing_rate"}})
    for unit in unit_list:
        for collection in _collections(unit):
            bucket = groups[collection]
            bucket["total_units"] += 1
            bucket["missing_title"] += int(not field_value(get(unit, "title") or metadata(unit).get("title")))
            bucket["missing_tags"] += int(not _tags(unit))
            bucket["missing_source"] += int(not field_value(get(unit, "source_id") or get(unit, "source_project") or metadata(unit).get("source_id")))
            bucket["missing_timestamps"] += int(not any(field_value(get(unit, key) or metadata(unit).get(key)) for key in ("created_at", "updated_at", "published_at")))
            bucket["blank_content"] += int(not field_value(get(unit, "content")))
    rows = []
    for collection in sorted(groups, key=sort_key):
        bucket = groups[collection]
        missing = sum(bucket[key] for key in ("missing_title", "missing_tags", "missing_source", "missing_timestamps", "blank_content"))
        total_checks = bucket["total_units"] * 5
        rows.append({"collection": collection, **bucket, "missing_rate": f"{(missing / total_checks) if total_checks else 0:.2f}"})
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _collections(unit: Mapping[str, Any] | object) -> list[str]:
    raw = get(unit, "collection") or get(unit, "collections") or metadata(unit).get("collection") or metadata(unit).get("collections")
    values = raw if isinstance(raw, list | tuple | set) else [raw]
    collections = [field_value(value) for value in values if field_value(value)]
    return collections or ["unassigned"]


def _tags(unit: Mapping[str, Any] | object) -> list[str]:
    raw = get(unit, "tags") or metadata(unit).get("tags")
    values = raw if isinstance(raw, list | tuple | set) else [raw]
    return [field_value(value) for value in values if field_value(value)]
