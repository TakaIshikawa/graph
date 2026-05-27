"""Summarize metadata value types across knowledge units."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_EXAMPLE_LIMIT = 3


def summarize_metadata_value_types(units: Iterable[Any], *, example_limit: int = DEFAULT_EXAMPLE_LIMIT) -> dict[str, Any]:
    """Return rows grouped by metadata key and coarse value type."""

    if example_limit < 0:
        raise ValueError("example_limit must be non-negative")

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    inspected_units = 0
    for unit in units:
        metadata = _metadata(unit)
        if not metadata:
            continue
        inspected_units += 1
        source = _string(_first(unit, metadata, ("source_project", "source")))
        for key, value in metadata.items():
            metadata_key = str(key)
            value_type = _value_type(value)
            group = groups.setdefault(
                (metadata_key, value_type),
                {
                    "metadata_key": metadata_key,
                    "value_type": value_type,
                    "unit_count": 0,
                    "non_empty_count": 0,
                    "example_values": [],
                    "sources": set(),
                },
            )
            group["unit_count"] += 1
            if _is_non_empty(value):
                group["non_empty_count"] += 1
                example = _example_value(value)
                if len(group["example_values"]) < example_limit and example not in group["example_values"]:
                    group["example_values"].append(example)
            if source is not None:
                group["sources"].add(source)

    rows = []
    for key in sorted(groups):
        group = groups[key]
        rows.append(
            {
                "metadata_key": group["metadata_key"],
                "value_type": group["value_type"],
                "unit_count": group["unit_count"],
                "non_empty_count": group["non_empty_count"],
                "example_values": group["example_values"],
                "sources": sorted(group["sources"]),
            }
        )
    return {"rows": rows, "row_count": len(rows), "unit_count": inspected_units}


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, Mapping):
        return "mapping"
    return "other"


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    if isinstance(value, (list, Mapping)) and not value:
        return False
    return True


def _example_value(value: Any) -> str:
    return repr(value)


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
