"""Duplicate member summary for collections."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

MEMBER_KEYS = ("members", "member_ids", "items", "unit_ids")


def summarize_collection_duplicate_members(collections: Iterable[Mapping[str, Any] | object]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any] | object]] = defaultdict(list)
    for collection in collections:
        grouped[(_source(collection), _collection_type(collection))].append(collection)

    rows: list[dict[str, Any]] = []
    for (source, collection_type), group in sorted(grouped.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))):
        duplicate_member_collection_count = 0
        duplicate_member_total = 0
        max_duplicate_member_count = 0
        sample_ids: set[str] = set()
        for collection in group:
            duplicates = {member_id: count - 1 for member_id, count in Counter(_member_ids(collection)).items() if count > 1}
            duplicate_count = sum(duplicates.values())
            if duplicate_count:
                duplicate_member_collection_count += 1
            duplicate_member_total += duplicate_count
            max_duplicate_member_count = max(max_duplicate_member_count, duplicate_count)
            sample_ids.update(duplicates)
        rows.append(
            {
                "source": source,
                "collection_type": collection_type,
                "collection_count": len(group),
                "duplicate_member_collection_count": duplicate_member_collection_count,
                "duplicate_member_total": duplicate_member_total,
                "max_duplicate_member_count": max_duplicate_member_count,
                "sample_duplicate_member_ids": sorted(sample_ids, key=_sort_key)[:5],
            }
        )
    return {"rows": rows, "collection_summaries": rows, "total_collections": sum(row["collection_count"] for row in rows)}


def _member_ids(collection: Mapping[str, Any] | object) -> list[str]:
    metadata = _metadata(collection)
    values: list[object] = []
    for key in MEMBER_KEYS:
        values.extend(_as_list(_get(collection, key)))
        values.extend(_as_list(metadata.get(key)))
    return [_member_id(value) for value in values if _member_id(value)]


def _member_id(value: object) -> str:
    if isinstance(value, Mapping):
        for key in ("id", "unit_id", "member_id", "source_id"):
            text = _text(value.get(key))
            if text:
                return text
    return _text(value)


def _as_list(value: object) -> list[object]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [] if value is None else [value]


def _source(collection: Mapping[str, Any] | object) -> str:
    metadata = _metadata(collection)
    return _text(_get(collection, "source")) or _text(_get(collection, "source_project")) or _text(metadata.get("source")) or "unknown"


def _collection_type(collection: Mapping[str, Any] | object) -> str:
    metadata = _metadata(collection)
    return _text(_get(collection, "type")) or _text(_get(collection, "collection_type")) or _text(metadata.get("type")) or "unknown"


def _metadata(value: Mapping[str, Any] | object) -> Mapping[str, Any]:
    metadata = _get(value, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: Mapping[str, Any] | object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _text(value: object) -> str:
    return "" if value is None else str(getattr(value, "value", value)).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
