"""Summarize collection members whose referenced records lack usable URLs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_COLLECTION_ID_KEYS = ("id", "collection_id", "source_id")
_MEMBER_KEYS = ("members", "member_ids", "unit_ids", "items")
_MEMBER_ID_KEYS = ("id", "unit_id", "member_id", "source_id")
_URL_KEYS = ("canonical_url", "url", "source_url", "href", "link")


def summarize_collection_member_missing_urls(collections: Iterable[Any], units: Iterable[Any] | None = None, sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    collection_list = list(collections)
    lookup = {_first(unit, metadata(unit), _MEMBER_ID_KEYS): unit for unit in units or []}
    rows: list[dict[str, Any]] = []
    for collection in collection_list:
        members = _members(collection)
        missing: list[str] = []
        for member in members:
            member_id = _member_id(member)
            source = member if isinstance(member, Mapping) else lookup.get(member_id, member)
            if not _usable_url(source):
                missing.append(member_id)
        if missing:
            total = len(members)
            rows.append(
                {
                    "collection_id": _collection_id(collection),
                    "member_count": total,
                    "missing_url_count": len(missing),
                    "url_coverage_ratio": round((total - len(missing)) / total, 4) if total else 1.0,
                    "sample_member_ids": sorted(missing, key=sort_key)[:limit],
                }
            )
    rows.sort(key=lambda row: sort_key(row["collection_id"]))
    return {"collection_count": len(collection_list), "affected_collection_count": len(rows), "collections": rows}


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


def _member_id(member: Any) -> str:
    if isinstance(member, Mapping):
        return _first(member, metadata(member), _MEMBER_ID_KEYS)
    return field_value(member)


def _collection_id(collection: Any) -> str:
    return _first(collection, metadata(collection), _COLLECTION_ID_KEYS)


def _usable_url(value: Any) -> bool:
    meta = metadata(value)
    for key in _URL_KEYS:
        raw = get(value, key)
        if isinstance(raw, str) and raw.strip():
            return True
        raw = meta.get(key)
        if isinstance(raw, str) and raw.strip():
            return True
    return False


def _first(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(item, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""
