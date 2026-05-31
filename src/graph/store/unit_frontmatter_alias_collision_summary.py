"""Summarize collisions in unit frontmatter aliases."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id


def summarize_unit_frontmatter_alias_collisions(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for unit in units:
        for alias in dict.fromkeys(_aliases(unit)):
            normalized = alias.casefold()
            grouped[normalized].append({"unit_id": unit_id(unit), "title": field_value(get(unit, "title") or metadata(unit).get("title")), "alias": alias})
            counts[normalized] += 1
    collisions = []
    for normalized, rows in grouped.items():
        unit_ids = {row["unit_id"] for row in rows}
        if len(unit_ids) > 1:
            collisions.append({"alias": sorted({row["alias"] for row in rows}, key=sort_key)[0], "normalized_alias": normalized, "unit_ids": sorted(unit_ids, key=sort_key), "titles": sorted({row["title"] for row in rows if row["title"]}, key=sort_key)})
    collisions.sort(key=lambda row: sort_key(row["normalized_alias"]))
    return {"alias_counts": [{"alias": key, "count": counts[key]} for key in sorted(counts, key=sort_key)], "collision_count": len(collisions), "collisions": collisions, "samples": collisions[:sample_limit]}


def _aliases(unit: Any) -> list[str]:
    meta = metadata(unit)
    values = _flatten(meta.get("aliases")) + _flatten(meta.get("alias"))
    return [field_value(value) for value in values if field_value(value)]


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _flatten(child)]
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in _flatten(child)]
    return [value]
