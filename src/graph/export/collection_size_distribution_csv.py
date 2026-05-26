"""CSV export for collection size distribution."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, write_csv

_FIELDNAMES = ["collection", "unit_count", "source_count", "min_content_length", "max_content_length", "average_content_length"]
_COLLECTION_KEYS = ("collection", "collections", "folder", "project", "notebook")
_UNASSIGNED = "unassigned"


def export_collection_size_distribution_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    buckets: dict[str, list[Mapping[str, Any] | object]] = defaultdict(list)
    for unit in unit_list:
        for collection in _collections(unit):
            buckets[collection].append(unit)
    rows = [_row(collection, grouped) for collection, grouped in sorted(buckets.items(), key=lambda item: sort_key(item[0]))]
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(collection: str, units: list[Mapping[str, Any] | object]) -> dict[str, str | int]:
    lengths = [len("" if get(unit, "content") is None else str(get(unit, "content"))) for unit in units]
    sources = {field_value(get(unit, "source_project") or metadata(unit).get("source") or metadata(unit).get("source_project")) for unit in units}
    return {
        "collection": collection,
        "unit_count": len(units),
        "source_count": len({source for source in sources if source}),
        "min_content_length": min(lengths, default=0),
        "max_content_length": max(lengths, default=0),
        "average_content_length": f"{(sum(lengths) / len(lengths)):.2f}" if lengths else "0.00",
    }


def _collections(unit: Mapping[str, Any] | object) -> list[str]:
    found: list[str] = []
    meta = metadata(unit)
    for key in _COLLECTION_KEYS:
        raw = get(unit, key)
        if raw in (None, "", []):
            raw = meta.get(key)
        found.extend(field_value(value) for value in flatten_values(raw) if field_value(value))
    return sorted(set(found), key=sort_key) or [_UNASSIGNED]
