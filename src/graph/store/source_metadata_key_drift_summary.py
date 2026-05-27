"""Summarize metadata key drift by source across chronological windows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, parse_datetime, sort_key

_SOURCE_KEYS = ("source", "source_id", "source_key")
_TIME_KEYS = ("created_at", "updated_at", "ingested_at")


def summarize_source_metadata_key_drift(units: Iterable[Any]) -> dict[str, Any]:
    groups: dict[str, list[tuple[Any, set[str]]]] = defaultdict(list)
    invalid_timestamp_count = 0
    for unit in units:
        meta = metadata(unit)
        source = _first(unit, meta, _SOURCE_KEYS) or "unknown"
        timestamp = next((parse_datetime(_first(unit, meta, (key,))) for key in _TIME_KEYS if _first(unit, meta, (key,))), None)
        if timestamp is None:
            invalid_timestamp_count += 1
        groups[source].append((timestamp, {field_value(key).casefold() for key in meta if field_value(key)}))

    rows = []
    for source in sorted(groups, key=sort_key):
        dated = sorted(groups[source], key=lambda item: (item[0] is None, item[0]))
        valid = [item for item in dated if item[0] is not None]
        if valid:
            midpoint = max(1, len(valid) // 2)
            earlier_items = valid[:midpoint]
            later_items = valid[midpoint:] or valid[:]
        else:
            earlier_items = []
            later_items = dated
        earlier = set().union(*(keys for _, keys in earlier_items)) if earlier_items else set()
        later = set().union(*(keys for _, keys in later_items)) if later_items else set()
        rows.append(
            {
                "source": source,
                "earlier_unit_count": len(earlier_items),
                "later_unit_count": len(later_items),
                "added_keys": sorted(later - earlier, key=sort_key),
                "removed_keys": sorted(earlier - later, key=sort_key),
                "stable_keys": sorted(earlier & later, key=sort_key),
            }
        )
    return {"source_count": len(rows), "invalid_timestamp_count": invalid_timestamp_count, "rows": rows}


def _first(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(item, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""
