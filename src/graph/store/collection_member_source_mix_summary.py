"""Summarize source-project mix for collection members."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_UNIT_ID_KEYS = ("id", "unit_id", "source_id")
_COLLECTION_ID_KEYS = ("id", "collection_id", "source_id")
_MEMBER_KEYS = ("members", "member_ids", "unit_ids", "items")
_NESTED_MEMBER_ID_KEYS = ("id", "unit_id", "member_id", "source_id")
_SOURCE_KEYS = ("source_project", "source")
_UNKNOWN_SOURCE = "unknown"


def summarize_collection_member_source_mix(collections: Iterable[Any], units: Iterable[Any]) -> dict[str, Any]:
    """Return per-collection source_project distributions for member IDs."""

    unit_sources = {
        unit_id: _source(unit)
        for unit in units
        for unit_id in [_first(unit, metadata(unit), _UNIT_ID_KEYS)]
        if unit_id
    }

    rows: list[dict[str, Any]] = []
    for collection in collections:
        meta = metadata(collection)
        member_ids = _member_ids(collection, meta)
        counts: Counter[str] = Counter()
        missing = 0
        for member_id in member_ids:
            source = unit_sources.get(member_id)
            if source is None:
                missing += 1
            else:
                counts[source] += 1
        rows.append(
            {
                "collection_id": _first(collection, meta, _COLLECTION_ID_KEYS),
                "total_members": len(member_ids),
                "matched_members": sum(counts.values()),
                "missing_members": missing,
                "dominant_source": _dominant_source(counts),
                "source_counts": dict(sorted(counts.items(), key=lambda item: sort_key(item[0]))),
                "mixed_source": len(counts) > 1,
            }
        )

    rows.sort(key=lambda row: sort_key(row["collection_id"]))
    return {"collection_count": len(rows), "rows": rows}


def _member_ids(item: Any, meta: Mapping[str, Any]) -> list[str]:
    for key in _MEMBER_KEYS:
        value = get(item, key)
        if value not in (None, ""):
            return _member_id_values(value)
        value = meta.get(key)
        if value not in (None, ""):
            return _member_id_values(value)
    return []


def _member_id_values(value: Any) -> list[str]:
    values = value if isinstance(value, list | tuple | set) else [value]
    result: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            text = _first(item, {}, _NESTED_MEMBER_ID_KEYS)
        else:
            text = field_value(item)
        if text:
            result.append(text)
    return result


def _source(unit: Any) -> str:
    return _first(unit, metadata(unit), _SOURCE_KEYS) or _UNKNOWN_SOURCE


def _first(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(item, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""


def _dominant_source(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))[0][0]
