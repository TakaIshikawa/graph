"""Summarize collection members that do not resolve to known units."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_UNIT_ID_KEYS = ("id", "unit_id", "source_id")
_COLLECTION_ID_KEYS = ("id", "collection_id", "source_id")
_TITLE_KEYS = ("title", "name")
_MEMBER_KEYS = ("members", "member_ids", "unit_ids", "items")
_NESTED_MEMBER_ID_KEYS = ("id", "unit_id", "source_id")


def summarize_collection_orphan_members(collections: Iterable[Any], units: Iterable[Any]) -> dict[str, Any]:
    """Return collections with declared member IDs that are absent from units."""

    unit_ids = {
        unit_id
        for unit in units
        for unit_id in [_string(_first(unit, _metadata(unit), _UNIT_ID_KEYS))]
        if unit_id is not None
    }

    rows = []
    total_collections = 0
    for collection in collections:
        total_collections += 1
        metadata = _metadata(collection)
        collection_id = _string(_first(collection, metadata, _COLLECTION_ID_KEYS))
        collection_title = _string(_first(collection, metadata, _TITLE_KEYS))
        members = _members(collection, metadata)
        resolved_count = 0
        missing: set[str] = set()
        for member_id in members:
            if member_id in unit_ids:
                resolved_count += 1
            else:
                missing.add(member_id)
        if not missing:
            continue
        rows.append(
            {
                "collection_id": collection_id,
                "collection_title": collection_title,
                "missing_member_count": len(missing),
                "missing_member_ids": sorted(missing),
                "resolved_member_count": resolved_count,
            }
        )

    rows.sort(key=lambda row: (row["collection_id"] or "", row["collection_title"] or ""))
    return {"rows": rows, "row_count": len(rows), "collection_count": total_collections}


def _members(item: Any, metadata: Mapping[str, Any]) -> list[str]:
    for key in _MEMBER_KEYS:
        value = _get(item, key)
        if value not in (None, ""):
            return _member_ids(value)
        value = metadata.get(key)
        if value not in (None, ""):
            return _member_ids(value)
    return []


def _member_ids(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        if isinstance(item, Mapping):
            nested = _first(item, {}, _NESTED_MEMBER_ID_KEYS)
            if nested not in (None, ""):
                result.append(str(nested))
        elif item not in (None, ""):
            result.append(str(item))
    return result


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _first(item: Any, metadata: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _get(item, key)
        if value not in (None, ""):
            return value
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
