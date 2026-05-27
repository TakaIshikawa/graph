"""Summarize duplicate external identifiers across units."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_IDENTIFIER_KEYS = ("source_id", "external_id", "external_url", "source_url", "url", "doi")
_UNKNOWN_SOURCE = "Unknown"


def summarize_unit_duplicate_external_ids(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    """Return duplicate identifier groups scoped by source and identifier key."""
    groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    missing: Counter[tuple[str, str]] = Counter()
    total_units = 0

    for unit in units:
        total_units += 1
        source = _source(unit)
        identifiers = _identifiers(unit)
        for key in _IDENTIFIER_KEYS:
            value = identifiers.get(key, "")
            if value:
                groups[(source, key, value)].append(unit)
            else:
                missing[(source, key)] += 1

    duplicate_values = []
    for (source, key, value), members in groups.items():
        if len(members) <= 1:
            continue
        ordered = sorted(members, key=lambda unit: sort_key(unit_id(unit)))
        duplicate_values.append(
            {
                "source": source,
                "identifier_key": key,
                "identifier_value": value,
                "unit_count": len(ordered),
                "sample_unit_ids": [unit_id(unit) for unit in ordered[:sample_limit]],
            }
        )
    duplicate_values.sort(
        key=lambda row: (
            sort_key(row["source"]),
            sort_key(row["identifier_key"]),
            -int(row["unit_count"]),
            sort_key(row["identifier_value"]),
        )
    )

    return {
        "total_units": total_units,
        "duplicate_group_count": len(duplicate_values),
        "duplicate_values": duplicate_values,
        "missing_identifier_counts": [
            {"source": source, "identifier_key": key, "missing_count": missing[(source, key)]}
            for source, key in sorted(missing, key=lambda item: (sort_key(item[0]), sort_key(item[1])))
        ],
    }


def _source(unit: Any) -> str:
    return field_value(get(unit, "source_project") or metadata(unit).get("source_project")) or _UNKNOWN_SOURCE


def _identifiers(unit: Any) -> dict[str, str]:
    values = {key: field_value(get(unit, key)) for key in _IDENTIFIER_KEYS}
    for key, value in metadata(unit).items():
        text_key = field_value(key).casefold().replace("-", "_").replace(" ", "_")
        if text_key in values and not values[text_key]:
            values[text_key] = field_value(value)
    return values
