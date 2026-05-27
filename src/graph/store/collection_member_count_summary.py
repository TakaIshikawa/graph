"""Summarize collection member count distributions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_SOURCE_KEYS = ("source_project", "source")
_TYPE_KEYS = ("collection_type", "type", "source_entity_type")
_MEMBER_KEYS = ("members", "member_ids", "unit_ids", "items")
_BUCKETS = ("empty", "singleton", "small", "medium", "large")


def summarize_collection_member_counts(collections: Iterable[Any]) -> dict[str, Any]:
    """Group collections by source/type and summarize member-count buckets."""

    groups: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    total_collections = 0
    for collection in collections:
        total_collections += 1
        metadata = _metadata(collection)
        source = _string(_first(collection, metadata, _SOURCE_KEYS))
        collection_type = _string(_first(collection, metadata, _TYPE_KEYS))
        count = len(_members(collection, metadata))
        group = groups.setdefault(
            (source, collection_type),
            {"source": source, "collection_type": collection_type, "collection_count": 0, "member_counts": []},
        )
        group["collection_count"] += 1
        group["member_counts"].append(count)

    rows = []
    for key in sorted(groups, key=lambda item: ((item[0] or ""), (item[1] or ""))):
        group = groups[key]
        counts = group["member_counts"]
        bucket_counts = {f"{bucket}_count": 0 for bucket in _BUCKETS}
        for count in counts:
            bucket_counts[f"{_bucket(count)}_count"] += 1
        rows.append(
            {
                "source": group["source"],
                "collection_type": group["collection_type"],
                "collection_count": group["collection_count"],
                **bucket_counts,
                "min_members": min(counts),
                "max_members": max(counts),
                "average_members": round(sum(counts) / len(counts), 2),
            }
        )
    return {"rows": rows, "row_count": len(rows), "collection_count": total_collections}


def _bucket(count: int) -> str:
    if count == 0:
        return "empty"
    if count == 1:
        return "singleton"
    if count <= 5:
        return "small"
    if count <= 20:
        return "medium"
    return "large"


def _members(item: Any, metadata: Mapping[str, Any]) -> list[Any]:
    for key in _MEMBER_KEYS:
        value = _get(item, key)
        if value not in (None, ""):
            return value if isinstance(value, list) else [value]
        value = metadata.get(key)
        if value not in (None, ""):
            return value if isinstance(value, list) else [value]
    return []


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
