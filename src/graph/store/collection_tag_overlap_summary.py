"""Summarize pairwise tag overlap between collections."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key


def summarize_collection_tag_overlap(collections: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    prepared = [_collection(collection) for collection in collections if _collection(collection)["tags"]]
    rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(prepared, key=lambda item: sort_key(item["collection_id"])), 2):
        shared = sorted(left["tags"] & right["tags"], key=sort_key)
        if not shared:
            continue
        union = left["tags"] | right["tags"]
        rows.append(
            {
                "collection_id_a": left["collection_id"],
                "collection_name_a": left["collection_name"],
                "collection_id_b": right["collection_id"],
                "collection_name_b": right["collection_name"],
                "shared_tag_count": len(shared),
                "jaccard": round(len(shared) / len(union), 4),
                "shared_tag_samples": shared[:limit],
            }
        )
    rows.sort(key=lambda row: (-float(row["jaccard"]), sort_key(row["collection_id_a"]), sort_key(row["collection_id_b"])))
    return {"collection_count": len(prepared), "overlaps": rows}


def _collection(collection: Any) -> dict[str, Any]:
    meta = metadata(collection)
    raw_tags = get(collection, "tags") or meta.get("tags") or []
    tags = {field_value(tag).casefold() for tag in raw_tags if field_value(tag)}
    return {
        "collection_id": field_value(get(collection, "id") or get(collection, "collection_id") or meta.get("id")),
        "collection_name": field_value(get(collection, "name") or get(collection, "title") or meta.get("name") or meta.get("title")),
        "tags": tags,
    }
