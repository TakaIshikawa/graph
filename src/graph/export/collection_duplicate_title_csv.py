"""CSV export for duplicate unit titles within collections."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["collection", "normalized_title", "duplicate_count", "unit_ids", "canonical_title", "source_ids"]


def export_collection_duplicate_title_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"titles": [], "unit_ids": set(), "source_ids": set()})
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        normalized = " ".join(title.casefold().split())
        if not normalized:
            continue
        for collection in _collections(unit):
            bucket = groups[(collection, normalized)]
            bucket["titles"].append(title)
            bucket["unit_ids"].add(unit_id(unit))
            source = field_value(get(unit, "source_id") or metadata(unit).get("source_id"))
            if source:
                bucket["source_ids"].add(source)
    rows = [
        {
            "collection": collection,
            "normalized_title": normalized,
            "duplicate_count": len(bucket["unit_ids"]),
            "unit_ids": "; ".join(sorted(bucket["unit_ids"], key=sort_key)),
            "canonical_title": bucket["titles"][0],
            "source_ids": "; ".join(sorted(bucket["source_ids"], key=sort_key)),
        }
        for (collection, normalized), bucket in groups.items()
        if len(bucket["unit_ids"]) > 1
    ]
    rows.sort(key=lambda row: (sort_key(row["collection"]), sort_key(row["normalized_title"])))
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
