"""Summarize tag density across collection members."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, sort_key

_COLLECTION_ID_KEYS = ("id", "collection_id", "source_id")
_MEMBER_KEYS = ("members", "items")
_TAG_KEYS = ("tags", "tag")


def summarize_collection_member_tag_density(collections: Iterable[Any], minimum_average_tags: float = 1.0) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sparse: list[dict[str, Any]] = []
    for collection in collections:
        members = _members(collection)
        tag_counts = [len(_tags(member)) for member in members]
        total = len(members)
        tagged = sum(1 for count in tag_counts if count > 0)
        average = round(sum(tag_counts) / total, 2) if total else 0.0
        row = {
            "collection_id": _collection_id(collection),
            "total_members": total,
            "tagged_members": tagged,
            "untagged_members": total - tagged,
            "average_tags_per_member": average,
            "max_tags_on_member": max(tag_counts, default=0),
        }
        rows.append(row)
        if average < minimum_average_tags:
            sparse.append(row)

    rows.sort(key=lambda row: sort_key(row["collection_id"]))
    sparse.sort(key=lambda row: sort_key(row["collection_id"]))
    return {"collection_count": len(rows), "collections": rows, "sparse_collections": sparse}


def _members(collection: Any) -> list[Any]:
    meta = metadata(collection)
    for key in _MEMBER_KEYS:
        value = get(collection, key)
        if value not in (None, ""):
            return list(value if isinstance(value, list | tuple | set) else [value])
        value = meta.get(key)
        if value not in (None, ""):
            return list(value if isinstance(value, list | tuple | set) else [value])
    return []


def _tags(member: Any) -> set[str]:
    meta = metadata(member)
    tags: set[str] = set()
    for key in _TAG_KEYS:
        value = get(member, key)
        if value not in (None, ""):
            tags.update(_tag_values(value))
        value = meta.get(key)
        if value not in (None, ""):
            tags.update(_tag_values(value))
    return tags


def _tag_values(value: Any) -> set[str]:
    if isinstance(value, str):
        values = value.split(",") if "," in value else [value]
    else:
        values = flatten_values(value)
    return {field_value(item).casefold() for item in values if field_value(item)}


def _collection_id(collection: Any) -> str:
    meta = metadata(collection)
    for key in _COLLECTION_ID_KEYS:
        value = field_value(get(collection, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""
