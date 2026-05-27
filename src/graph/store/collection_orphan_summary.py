"""Summarize empty collections and dangling collection references."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_COLLECTION_ID_KEYS = ("id", "collection_id", "source_id")
_MEMBER_KEYS = ("members", "member_ids", "unit_ids", "items")
_UNIT_COLLECTION_KEYS = ("collection_id", "collection_ids", "collections")
_NESTED_ID_KEYS = ("id", "unit_id", "member_id", "source_id")


def summarize_collection_orphans(collections: Iterable[Any], units: Iterable[Any] | None = None, *, sample_limit: int = 5) -> dict[str, Any]:
    collection_rows = list(collections)
    collection_ids = [_first(collection, metadata(collection), _COLLECTION_ID_KEYS) for collection in collection_rows]
    known = {collection_id for collection_id in collection_ids if collection_id}
    empty_ids: list[str] = []
    for index, collection in enumerate(collection_rows):
        meta = metadata(collection)
        members = _ids(_first_raw(collection, meta, _MEMBER_KEYS))
        if not members:
            empty_ids.append(_first(collection, meta, _COLLECTION_ID_KEYS) or str(index))

    dangling: list[dict[str, str]] = []
    for unit in units or []:
        uid = unit_id(unit)
        meta = metadata(unit)
        for collection_id in _ids(_first_raw(unit, meta, _UNIT_COLLECTION_KEYS)):
            if collection_id and collection_id not in known:
                dangling.append({"unit_id": uid, "collection_id": collection_id})
    dangling.sort(key=lambda row: (sort_key(row["collection_id"]), sort_key(row["unit_id"])))
    return {
        "total_collections": len(collection_rows),
        "empty_collection_count": len(empty_ids),
        "empty_collection_ids": sorted(empty_ids, key=sort_key)[: max(0, sample_limit)],
        "dangling_reference_count": len(dangling),
        "dangling_references": dangling[: max(0, sample_limit)],
    }


def _ids(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    values = value if isinstance(value, list | tuple | set) else [value]
    result: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            text = _first(item, {}, _NESTED_ID_KEYS)
        else:
            text = field_value(item)
        if text:
            result.append(text)
    return result


def _first_raw(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = get(item, key)
        if value not in (None, ""):
            return value
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return None


def _first(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(item, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""
