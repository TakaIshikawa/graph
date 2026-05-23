"""CSV export for tag coverage by collection."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, normalized_key, render_csv, sort_key, unit_id, write_csv
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["collection", "total_units", "tagged_units", "untagged_units", "coverage_ratio", "top_tags", "sample_unit_ids"]
_COLLECTION_KEYS = {"collection", "collection_id", "collection_name", "collections", "project", "folder", "list", "notebook"}
_TAG_KEYS = {"tag", "tags", "labels", "keywords"}
_UNASSIGNED = "unassigned"
_SAMPLE_LIMIT = 5


def export_collection_tag_coverage_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write collection-level tag coverage metrics."""
    unit_list = list(units)
    rows = _coverage_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _coverage_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "tagged": 0, "tags": Counter(), "unit_ids": []})
    for unit in units:
        tags = _unit_tags(unit)
        for collection in _unit_collections(unit) or [_UNASSIGNED]:
            group = groups[collection]
            group["total"] += 1
            if tags:
                group["tagged"] += 1
                group["tags"].update(tags)
            if unit_id(unit):
                group["unit_ids"].append(unit_id(unit))

    rows: list[dict[str, str | int]] = []
    for collection in sorted(groups, key=sort_key):
        group = groups[collection]
        untagged = group["total"] - group["tagged"]
        rows.append(
            {
                "collection": collection,
                "total_units": group["total"],
                "tagged_units": group["tagged"],
                "untagged_units": untagged,
                "coverage_ratio": f"{(group['tagged'] / group['total']):.2f}" if group["total"] else "0.00",
                "top_tags": "; ".join(f"{tag}:{count}" for tag, count in sorted(group["tags"].items(), key=lambda item: (-item[1], sort_key(item[0])))[:_SAMPLE_LIMIT]),
                "sample_unit_ids": "; ".join(sorted(set(group["unit_ids"]), key=sort_key)[:_SAMPLE_LIMIT]),
            }
        )
    return rows


def _unit_collections(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    values: set[str] = set()
    for key in _COLLECTION_KEYS:
        text = field_value(get(unit, key))
        if text:
            values.add(text)
    for raw_key, value in metadata(unit).items():
        if normalized_key(raw_key) in _COLLECTION_KEYS:
            values.update(field_value(item) for item in flatten_values(value) if field_value(item))
    return sorted(values, key=sort_key)


def _unit_tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    values: set[str] = {field_value(tag) for tag in flatten_values(get(unit, "tags")) if field_value(tag)}
    for raw_key, value in metadata(unit).items():
        if normalized_key(raw_key) in _TAG_KEYS:
            values.update(field_value(item) for item in flatten_values(value) if field_value(item))
    return sorted(values, key=sort_key)

