"""Summarize collections with missing or empty descriptions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_ID_KEYS = ("id", "collection_id", "source_id")
_DESCRIPTION_KEYS = ("description", "summary")
_GROUP_KEYS = ("source", "source_id", "type", "collection_type")


def summarize_collection_empty_descriptions(collections: Iterable[Any]) -> dict[str, Any]:
    total = empty = described = 0
    affected: list[str] = []
    groups: Counter[str] = Counter()
    for index, collection in enumerate(collections):
        total += 1
        meta = metadata(collection)
        description = _first(collection, meta, _DESCRIPTION_KEYS, preserve_whitespace=True)
        if description.strip():
            described += 1
            continue
        empty += 1
        affected.append(_first(collection, meta, _ID_KEYS) or str(index))
        groups[_first(collection, meta, _GROUP_KEYS) or "unknown"] += 1
    return {
        "total_collections": total,
        "empty_description_count": empty,
        "described_count": described,
        "affected_collection_ids": sorted(affected, key=sort_key),
        "counts_by_source": [{"source": key, "count": groups[key]} for key in sorted(groups, key=sort_key)],
    }


def _first(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...], *, preserve_whitespace: bool = False) -> str:
    for key in keys:
        raw = get(item, key)
        value = "" if raw is None else str(raw) if preserve_whitespace else field_value(raw)
        if value:
            return value
        raw = meta.get(key)
        value = "" if raw is None else str(raw) if preserve_whitespace else field_value(raw)
        if value:
            return value
    return ""
