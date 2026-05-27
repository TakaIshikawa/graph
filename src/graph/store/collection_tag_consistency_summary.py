"""Summarize consistency between collection tags and member unit tags."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_COLLECTION_ID_KEYS = ("id", "collection_id", "source_id")
_MEMBER_KEYS = ("members", "member_ids", "unit_ids", "items")
_TAG_KEYS = ("tags", "tag_names")


def summarize_collection_tag_consistency(collections: Iterable[Any], units: Iterable[Any]) -> dict[str, Any]:
    unit_tags = {unit_id(unit): _tags(unit) for unit in units if unit_id(unit)}
    rows = []
    total = 0
    for index, collection in enumerate(collections):
        total += 1
        meta = metadata(collection)
        cid = _first(collection, meta, _COLLECTION_ID_KEYS) or str(index)
        collection_tags = _tags(collection)
        member_ids = _members(collection, meta)
        member_tags = set().union(*(unit_tags.get(member_id, set()) for member_id in member_ids)) if member_ids else set()
        overlap = collection_tags & member_tags
        ratio = len(overlap) / len(collection_tags) if collection_tags else 1.0
        rows.append(
            {
                "collection_id": cid,
                "member_count": len(member_ids),
                "consistency_ratio": ratio,
                "missing_collection_tags": sorted(collection_tags - member_tags, key=sort_key),
                "member_only_tags": sorted(member_tags - collection_tags, key=sort_key),
            }
        )
    rows.sort(key=lambda row: sort_key(row["collection_id"]))
    return {"total_collections": total, "rows": rows}


def _tags(item: Any) -> set[str]:
    meta = metadata(item)
    value = next((get(item, key) for key in _TAG_KEYS if get(item, key) not in (None, "")), None)
    if value is None:
        value = next((meta.get(key) for key in _TAG_KEYS if meta.get(key) not in (None, "")), None)
    values = value if isinstance(value, list | tuple | set) else [value] if value not in (None, "") else []
    return {field_value(value).casefold() for value in values if field_value(value)}


def _members(item: Any, meta: Mapping[str, Any]) -> list[str]:
    value = next((get(item, key) for key in _MEMBER_KEYS if get(item, key) not in (None, "")), None)
    if value is None:
        value = next((meta.get(key) for key in _MEMBER_KEYS if meta.get(key) not in (None, "")), None)
    values = value if isinstance(value, list | tuple | set) else [value] if value not in (None, "") else []
    result = []
    for member in values:
        if isinstance(member, Mapping):
            result.append(_first(member, {}, ("id", "unit_id", "source_id")))
        else:
            result.append(field_value(member))
    return [member for member in result if member]


def _first(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(item, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""
