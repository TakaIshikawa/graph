"""Cross-source coverage summary for collection members."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

MEMBER_KEYS = ("members", "member_ids", "items", "unit_ids")


def summarize_collection_cross_source_coverage(
    collections: Iterable[Mapping[str, Any] | object], units: Iterable[Mapping[str, Any] | object]
) -> dict[str, Any]:
    unit_sources = {_unit_id(unit): _source(unit) for unit in units if _unit_id(unit)}
    collection_summaries: list[dict[str, Any]] = []
    for collection in collections:
        member_ids = _member_ids(collection)
        counts = Counter(unit_sources[member_id] for member_id in member_ids if member_id in unit_sources)
        missing_count = sum(1 for member_id in member_ids if member_id not in unit_sources)
        dominant_source = _dominant_source(counts)
        distinct_source_count = len(counts)
        collection_summaries.append(
            {
                "collection_id": _collection_id(collection),
                "source": _source(collection),
                "collection_type": _collection_type(collection),
                "member_count": len(member_ids),
                "known_member_count": sum(counts.values()),
                "missing_member_count": missing_count,
                "source_counts": [{"source": source, "count": count} for source, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))],
                "dominant_source": dominant_source,
                "is_single_source": distinct_source_count == 1 and missing_count == 0,
                "is_cross_source": distinct_source_count > 1,
            }
        )
    collection_summaries.sort(key=lambda row: (_sort_key(row["source"]), _sort_key(row["collection_type"]), _sort_key(row["collection_id"])))
    return {
        "total_collections": len(collection_summaries),
        "cross_source_collection_count": sum(1 for row in collection_summaries if row["is_cross_source"]),
        "collection_summaries": collection_summaries,
    }


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


def _unit_id(unit: Mapping[str, Any] | object) -> str:
    for key in ("id", "unit_id", "source_id"):
        text = _text(_get(unit, key))
        if text:
            return text
    return ""


def _dominant_source(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))[0][0]


def _collection_id(collection: Mapping[str, Any] | object) -> str:
    return _text(_get(collection, "id")) or _text(_get(collection, "collection_id")) or _text(_metadata(collection).get("id"))


def _source(value: Mapping[str, Any] | object) -> str:
    metadata = _metadata(value)
    return _text(_get(value, "source_project")) or _text(_get(value, "source")) or _text(metadata.get("source_project")) or _text(metadata.get("source")) or "unknown"


def _collection_type(collection: Mapping[str, Any] | object) -> str:
    metadata = _metadata(collection)
    return _text(_get(collection, "type")) or _text(_get(collection, "collection_type")) or _text(metadata.get("type")) or "unknown"


def _as_list(value: object) -> list[object]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [] if value is None else [value]


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
