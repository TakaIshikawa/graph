"""Summarize duplicate canonical URLs among collection members."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from graph.export._report_csv import field_value, get, metadata, sort_key

_COLLECTION_ID_KEYS = ("id", "collection_id", "source_id")
_MEMBER_KEYS = ("members", "member_ids", "unit_ids", "items")
_MEMBER_ID_KEYS = ("id", "unit_id", "member_id", "source_id")
_URL_KEYS = ("canonical_url", "url", "source_url", "href", "link")


def summarize_collection_member_duplicate_urls(collections: Iterable[Any], units: Iterable[Any] | None = None, sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    collection_list = list(collections)
    lookup = {_first(unit, metadata(unit), _MEMBER_ID_KEYS): unit for unit in units or []}
    rows: list[dict[str, Any]] = []

    for collection in collection_list:
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for member in _members(collection):
            member_id = _member_id(member)
            source = member if isinstance(member, Mapping) else lookup.get(member_id, member)
            url = _canonical_url(source)
            if url:
                grouped[url].append(member_id)
        duplicates = [
            {"url": url, "member_ids": ids}
            for url, ids in grouped.items()
            if len(set(ids)) > 1
        ]
        duplicates.sort(key=lambda row: sort_key(row["url"]))
        if duplicates:
            rows.append({"collection_id": _collection_id(collection), "duplicate_url_count": len(duplicates), "examples": duplicates[:limit]})

    rows.sort(key=lambda row: sort_key(row["collection_id"]))
    return {"collection_count": len(collection_list), "collections": rows}


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


def _canonical_url(value: Any) -> str:
    meta = metadata(value)
    for key in _URL_KEYS:
        raw = field_value(get(value, key)) or field_value(meta.get(key))
        if raw:
            return _normalize_url(raw)
    return ""


def _normalize_url(raw: str) -> str:
    try:
        parts = urlsplit(raw)
    except ValueError:
        return field_value(raw)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), path, parts.query, ""))


def _first(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(item, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""
