"""CSV export for pairwise collection tag overlap."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, write_csv

_FIELDNAMES = ["collection_id_a", "collection_id_b", "shared_tag_count", "shared_tags", "jaccard_similarity", "only_a_count", "only_b_count"]


def export_collection_tag_overlap_csv(collections: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    collection_list = [_collection(collection) for collection in collections]
    rows: list[dict[str, str | int]] = []
    for left, right in combinations(sorted(collection_list, key=lambda item: sort_key(item["id"])), 2):
        shared_keys = sorted(left["tags"] & right["tags"], key=sort_key)
        union = left["tags"] | right["tags"]
        rows.append(
            {
                "collection_id_a": left["id"],
                "collection_id_b": right["id"],
                "shared_tag_count": len(shared_keys),
                "shared_tags": "; ".join(left["labels"].get(key, key) for key in shared_keys),
                "jaccard_similarity": f"{(len(shared_keys) / len(union)):.4f}" if union else "0.0000",
                "only_a_count": len(left["tags"] - right["tags"]),
                "only_b_count": len(right["tags"] - left["tags"]),
            }
        )
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "collection_count": len(collection_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _collection(collection: Mapping[str, Any] | object) -> dict[str, Any]:
    raw_tags = get(collection, "tags") or metadata(collection).get("tags") or []
    labels: dict[str, str] = {}
    for tag in raw_tags:
        label = field_value(tag)
        if label:
            labels.setdefault(label.casefold(), label)
    return {"id": field_value(get(collection, "id") or get(collection, "collection_id") or metadata(collection).get("id")), "tags": set(labels), "labels": labels}
