"""Summarize empty top-level unit metadata values."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, metadata, sort_key, unit_id


def summarize_unit_metadata_empty_values(units: Iterable[Any]) -> dict[str, Any]:
    key_units: dict[str, list[str]] = defaultdict(list)
    total_units = 0
    affected_unit_ids: set[str] = set()
    for unit in units:
        total_units += 1
        uid = unit_id(unit)
        for key, value in metadata(unit).items():
            key_text = field_value(key)
            if key_text and _is_empty(value):
                key_units[key_text].append(uid)
                affected_unit_ids.add(uid)

    rows = [
        {"key": key, "empty_count": len(ids), "unit_ids": sorted(ids, key=sort_key)}
        for key, ids in sorted(key_units.items(), key=lambda item: sort_key(item[0]))
    ]
    return {
        "total_units": total_units,
        "units_with_empty_metadata_values": len(affected_unit_ids),
        "key_counts": {row["key"]: row["empty_count"] for row in rows},
        "affected_unit_ids": sorted(affected_unit_ids, key=sort_key),
        "rows": rows,
    }


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, Mapping | list | tuple | set):
        return len(value) == 0
    return False
