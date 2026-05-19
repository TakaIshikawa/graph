"""CSV export for unit location metadata coverage."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "source_entity_type",
    "unit_count",
    "coordinate_count",
    "named_place_count",
    "country_count",
    "missing_location_count",
    "representative_unit_ids",
]
_UNKNOWN = "Unknown"
_LAT_KEYS = ("latitude", "lat")
_LON_KEYS = ("longitude", "lon", "lng")
_PLACE_KEYS = ("place", "city", "region", "geohash")
_COUNTRY_KEYS = ("country",)
_LOCATION_KEYS = set(_LAT_KEYS + _LON_KEYS + _PLACE_KEYS + _COUNTRY_KEYS)
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_location_coverage_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write location coverage grouped by source project and entity type."""
    unit_list = list(units)
    rows = _coverage_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    return _write_output(
        path,
        text,
        {
            "unit_count": len(unit_list),
            "rows_exported": len(rows),
        },
    )


def _coverage_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "unit_ids": set(),
            "coordinate_ids": set(),
            "named_place_ids": set(),
            "country_ids": set(),
            "missing_ids": set(),
        }
    )

    for unit in units:
        unit_id = _unit_id(unit)
        group = groups[(_unit_source(unit), _unit_source_type(unit))]
        if unit_id:
            group["unit_ids"].add(unit_id)

        has_coordinates = _has_coordinate_pair(unit)
        has_country = _has_any_value(unit, _COUNTRY_KEYS)
        has_named_place = _has_any_value(unit, _PLACE_KEYS)
        has_location_signal = has_coordinates or has_country or has_named_place or _has_any_value(unit, _LOCATION_KEYS)

        if unit_id and has_coordinates:
            group["coordinate_ids"].add(unit_id)
        if unit_id and has_named_place and not has_coordinates:
            group["named_place_ids"].add(unit_id)
        if unit_id and has_country:
            group["country_ids"].add(unit_id)
        if unit_id and not has_location_signal:
            group["missing_ids"].add(unit_id)

    rows: list[dict[str, str | int]] = []
    for (source_project, source_entity_type), group in groups.items():
        rows.append(
            {
                "source_project": source_project,
                "source_entity_type": source_entity_type,
                "unit_count": len(group["unit_ids"]),
                "coordinate_count": len(group["coordinate_ids"]),
                "named_place_count": len(group["named_place_ids"]),
                "country_count": len(group["country_ids"]),
                "missing_location_count": len(group["missing_ids"]),
                "representative_unit_ids": _joined(group["unit_ids"]),
            }
        )

    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["source_entity_type"])))


def _has_coordinate_pair(unit: KnowledgeUnit | Mapping[str, Any]) -> bool:
    metadata = _metadata(unit)
    latitude = _first_numeric(metadata, _LAT_KEYS)
    longitude = _first_numeric(metadata, _LON_KEYS)
    return latitude is not None and longitude is not None


def _first_numeric(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _number(metadata.get(key))
        if value is not None:
            return value
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(_field_value(value))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _has_any_value(unit: KnowledgeUnit | Mapping[str, Any], keys: Iterable[str]) -> bool:
    metadata = _metadata(unit)
    return any(_field_value(metadata.get(key)) for key in keys)


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _unit_source(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "source_project")) or _UNKNOWN


def _unit_source_type(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "source_entity_type")) or _UNKNOWN


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _joined(values: Iterable[object]) -> str:
    return "; ".join(sorted({_field_value(value) for value in values if _field_value(value)}, key=_sort_key))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_output(path: str | Path | Any, text: str, stats: dict[str, Any]) -> dict[str, Any]:
    if hasattr(path, "write") and not isinstance(path, str | Path):
        written = path.write(text)
        stats["bytes_written"] = len(text.encode("utf-8")) if written is None else written
        return stats

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    stats["path"] = str(output_path)
    stats["bytes_written"] = output_path.stat().st_size
    return stats


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
