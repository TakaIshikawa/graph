"""CSV export for collection tag drift between early and late units."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, parse_datetime, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["collection_id", "tag", "early_count", "late_count", "delta", "drift_status", "unit_ids"]
_TAG_KEYS = {"tag", "tags", "label", "labels", "keyword", "keywords"}


def export_collection_tag_drift_csv(
    collections: Iterable[Mapping[str, Any] | object],
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    collection_list = list(collections)
    unit_list = list(units)
    rows = _rows(collection_list, unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "collection_count": len(collection_list), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(collections: list[Mapping[str, Any] | object], units: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    collection_ids = [_collection_id(collection) for collection in collections]
    members: dict[str, list[Mapping[str, Any] | object]] = {collection_id: [] for collection_id in collection_ids}
    explicit_members = _explicit_members(collections)
    units_by_id = {unit_id(unit): unit for unit in units}

    for collection_id, unit_ids in explicit_members.items():
        members.setdefault(collection_id, [])
        members[collection_id].extend(units_by_id[item] for item in unit_ids if item in units_by_id)

    for unit in units:
        for collection_id in _unit_collection_ids(unit):
            members.setdefault(collection_id, []).append(unit)

    rows: list[dict[str, str | int]] = []
    for collection_id, raw_members in members.items():
        ordered_units = _dedupe_units(raw_members)
        midpoint = (len(ordered_units) + 1) // 2
        early_units = ordered_units[:midpoint]
        late_units = ordered_units[midpoint:]
        early_counts = Counter(tag for unit in early_units for tag in set(_tags(unit)))
        late_counts = Counter(tag for unit in late_units for tag in set(_tags(unit)))
        tag_units = _tag_units(ordered_units)
        for tag in sorted(set(early_counts) | set(late_counts), key=sort_key):
            early = early_counts[tag]
            late = late_counts[tag]
            rows.append(
                {
                    "collection_id": collection_id,
                    "tag": tag,
                    "early_count": early,
                    "late_count": late,
                    "delta": late - early,
                    "drift_status": _status(early, late),
                    "unit_ids": "; ".join(sorted(tag_units[tag], key=sort_key)),
                }
            )
    return sorted(rows, key=lambda row: (sort_key(row["collection_id"]), sort_key(row["tag"])))


def _explicit_members(collections: list[Mapping[str, Any] | object]) -> dict[str, list[str]]:
    members: dict[str, list[str]] = {}
    for collection in collections:
        collection_id = _collection_id(collection)
        values = [get(collection, "unit_ids"), metadata(collection).get("unit_ids")]
        unit_ids = [field_value(item) for value in values for item in flatten_values(value) if field_value(item)]
        members[collection_id] = unit_ids
    return members


def _dedupe_units(units: list[Mapping[str, Any] | object]) -> list[Mapping[str, Any] | object]:
    by_id = {unit_id(unit): unit for unit in units}
    return sorted(by_id.values(), key=lambda unit: (_unit_order_key(unit), sort_key(unit_id(unit))))


def _unit_order_key(unit: Mapping[str, Any] | object) -> tuple[int, str]:
    date = parse_datetime(get(unit, "updated_at") or get(unit, "created_at") or metadata(unit).get("updated_at") or metadata(unit).get("created_at") or metadata(unit).get("date"))
    if date:
        return (0, date.isoformat())
    order = field_value(get(unit, "order") or metadata(unit).get("order") or metadata(unit).get("position"))
    return (1, order)


def _tag_units(units: list[Mapping[str, Any] | object]) -> dict[str, set[str]]:
    buckets: dict[str, set[str]] = defaultdict(set)
    for unit in units:
        for tag in set(_tags(unit)):
            buckets[tag].add(unit_id(unit))
    return buckets


def _tags(unit: Mapping[str, Any] | object) -> list[str]:
    tags: list[str] = []
    containers = [unit] if isinstance(unit, Mapping) else []
    containers.append(metadata(unit))
    for container in containers:
        for key, value in container.items():
            if field_value(key).casefold() in _TAG_KEYS:
                tags.extend(field_value(item).casefold() for item in flatten_values(value) if field_value(item))
    return tags


def _unit_collection_ids(unit: Mapping[str, Any] | object) -> list[str]:
    values = [get(unit, "collection_id"), get(unit, "collection_ids"), metadata(unit).get("collection_id"), metadata(unit).get("collection_ids"), metadata(unit).get("collection")]
    return sorted({field_value(item) for value in values for item in flatten_values(value) if field_value(item)}, key=sort_key)


def _collection_id(collection: Mapping[str, Any] | object) -> str:
    return field_value(get(collection, "id") or get(collection, "collection_id") or metadata(collection).get("id"))


def _status(early: int, late: int) -> str:
    if early and not late:
        return "fading"
    if late and not early:
        return "emerging"
    if early == late:
        return "stable"
    return "one_sided"
