"""CSV export for units with missing or dangling collection references."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "claimed_collection_id", "orphan_reason", "source_id"]
_COLLECTION_KEYS = ("collection_id", "collection", "collection_ref")


def export_collection_orphan_unit_csv(
    collections: Iterable[Mapping[str, Any] | object],
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    collection_ids = {unit_id(collection) for collection in collections if unit_id(collection)}
    unit_list = list(units)
    rows = []
    for unit in unit_list:
        claimed = _collection_id(unit)
        if claimed and claimed in collection_ids:
            continue
        rows.append(
            {
                "unit_id": unit_id(unit),
                "claimed_collection_id": claimed,
                "orphan_reason": "dangling_collection" if claimed else "missing_collection",
                "source_id": field_value(get(unit, "source_id")) or field_value(metadata(unit).get("source_id")),
            }
        )
    rows.sort(key=lambda row: (sort_key(row["orphan_reason"]), sort_key(row["unit_id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _collection_id(unit: Any) -> str:
    meta = metadata(unit)
    for key in _COLLECTION_KEYS:
        text = field_value(get(unit, key)) or field_value(meta.get(key))
        if text:
            return text
    return ""
