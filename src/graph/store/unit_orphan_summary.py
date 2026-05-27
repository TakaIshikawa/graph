"""Summarize knowledge units without incoming or outgoing relations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_UNIT_ID_KEYS = ("id", "unit_id", "source_id")
_SOURCE_KEYS = ("source_project", "source")
_ENTITY_TYPE_KEYS = ("source_entity_type", "entity_type", "type")
_FROM_KEYS = ("from_unit_id", "source_unit_id", "from_id")
_TO_KEYS = ("to_unit_id", "target_unit_id", "to_id")


def summarize_unit_orphans(units: Iterable[Any], edges: Iterable[Any]) -> dict[str, Any]:
    """Return units grouped by source and entity type when they have no edge endpoints."""

    connected: set[str] = set()
    for edge in edges:
        metadata = _metadata(edge)
        for keys in (_FROM_KEYS, _TO_KEYS):
            endpoint = _string(_first(edge, metadata, keys))
            if endpoint is not None:
                connected.add(endpoint)

    groups: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    total_units = 0
    for unit in units:
        total_units += 1
        metadata = _metadata(unit)
        unit_id = _string(_first(unit, metadata, _UNIT_ID_KEYS))
        source = _string(_first(unit, metadata, _SOURCE_KEYS))
        entity_type = _string(_first(unit, metadata, _ENTITY_TYPE_KEYS))
        key = (source, entity_type)
        group = groups.setdefault(key, {"source": source, "entity_type": entity_type, "total_units": 0, "unit_ids": []})
        group["total_units"] += 1
        if unit_id is not None and unit_id not in connected:
            group["unit_ids"].append(unit_id)

    rows = []
    for key in sorted(groups, key=lambda item: ((item[0] or ""), (item[1] or ""))):
        group = groups[key]
        unit_ids = sorted(group["unit_ids"])
        if not unit_ids:
            continue
        rows.append(
            {
                "source": group["source"],
                "entity_type": group["entity_type"],
                "orphan_count": len(unit_ids),
                "unit_ids": unit_ids,
                "total_units": group["total_units"],
            }
        )
    return {"rows": rows, "row_count": len(rows), "unit_count": total_units}


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
