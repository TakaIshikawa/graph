"""Summarize metadata key drift across collection groups."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_SOURCE_KEYS = ("source", "source_project", "source_id", "source_key")
_TYPE_KEYS = ("collection_type", "type", "kind")


def summarize_collection_metadata_drift(collections: Iterable[Any]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[set[str]]] = defaultdict(list)
    total = 0
    for collection in collections:
        total += 1
        groups[(_first(collection, _SOURCE_KEYS), _first(collection, _TYPE_KEYS))].append(_metadata_keys(collection))

    rows = []
    for (source, collection_type) in sorted(groups, key=lambda item: (sort_key(item[0]), sort_key(item[1]))):
        schemas = groups[(source, collection_type)]
        key_counts = Counter(key for keys in schemas for key in keys)
        variant_counts = Counter(tuple(sorted(keys, key=sort_key)) for keys in schemas)
        rows.append(
            {
                "source": source,
                "collection_type": collection_type,
                "collection_count": len(schemas),
                "distinct_metadata_key_count": len(key_counts),
                "common_keys": [
                    {"key": key, "count": key_counts[key]}
                    for key in sorted((key for key in key_counts if key_counts[key] == len(schemas)), key=sort_key)
                ],
                "rare_keys": [
                    {"key": key, "count": key_counts[key]} for key in sorted((key for key in key_counts if key_counts[key] == 1), key=sort_key)
                ],
                "schema_variants": [
                    {"keys": list(keys), "count": variant_counts[keys]} for keys in sorted(variant_counts, key=lambda keys: (len(keys), tuple(sort_key(key) for key in keys)))
                ],
            }
        )
    return {"total_collections": total, "rows": rows}


def _metadata_keys(collection: Any) -> set[str]:
    return {field_value(key).casefold() for key in metadata(collection) if field_value(key)}


def _first(collection: Any, keys: tuple[str, ...]) -> str:
    meta = metadata(collection)
    for key in keys:
        value = field_value(get(collection, key)) or field_value(meta.get(key))
        if value:
            return value
    return "unknown"
