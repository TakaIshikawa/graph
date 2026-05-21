"""CSV export for coarse geospatial unit clusters."""

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
    "cluster_key",
    "source_project",
    "unit_count",
    "representative_unit_ids",
    "centroid_latitude",
    "centroid_longitude",
    "min_latitude",
    "max_latitude",
    "min_longitude",
    "max_longitude",
]
_LAT_KEYS = ("latitude", "lat")
_LON_KEYS = ("longitude", "lon", "lng")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_geospatial_cluster_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    precision: int = 1,
) -> str | dict[str, Any]:
    """Return or write rounded coordinate clusters."""
    unit_list = list(units)
    rows = _cluster_rows(unit_list, precision)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _cluster_rows(units: list[KnowledgeUnit | Mapping[str, Any]], precision: int) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[tuple[str, float, float]]] = defaultdict(list)
    for unit in units:
        coordinate = _coordinate(unit)
        if coordinate is None:
            continue
        lat, lon = coordinate
        source = _field_value(_get(unit, "source_project")) or "Unknown"
        key = f"{round(lat, precision):.{precision}f},{round(lon, precision):.{precision}f}"
        groups[(source, key)].append((_unit_id(unit), lat, lon))

    rows: list[dict[str, str | int]] = []
    for (source, key), values in groups.items():
        lats = [value[1] for value in values]
        lons = [value[2] for value in values]
        rows.append(
            {
                "cluster_key": key,
                "source_project": source,
                "unit_count": len(values),
                "representative_unit_ids": "; ".join(sorted({value[0] for value in values if value[0]}, key=_sort_key)[:5]),
                "centroid_latitude": f"{sum(lats) / len(lats):.6f}",
                "centroid_longitude": f"{sum(lons) / len(lons):.6f}",
                "min_latitude": f"{min(lats):.6f}",
                "max_latitude": f"{max(lats):.6f}",
                "min_longitude": f"{min(lons):.6f}",
                "max_longitude": f"{max(lons):.6f}",
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["cluster_key"])))


def _coordinate(unit: KnowledgeUnit | Mapping[str, Any]) -> tuple[float, float] | None:
    metadata = _metadata(unit)
    lat = _first_number(unit, metadata, _LAT_KEYS)
    lon = _first_number(unit, metadata, _LON_KEYS)
    if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None
    return lat, lon


def _first_number(unit: KnowledgeUnit | Mapping[str, Any], metadata: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        for value in (_get(unit, key), _casefold_get(metadata, key)):
            number = _number(value)
            if number is not None:
                return number
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(_field_value(value))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)

