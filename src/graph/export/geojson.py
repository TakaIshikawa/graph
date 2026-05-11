"""GeoJSON export helpers for location-tagged knowledge units."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit

_COORDINATE_KEYS = (
    ("lat", "lon"),
    ("lat", "lng"),
    ("latitude", "longitude"),
    ("geo.lat", "geo.lon"),
    ("geo.latitude", "geo.longitude"),
    ("location.latitude", "location.longitude"),
)


def export_units_to_geojson(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    include_summary: bool = False,
) -> str:
    """Return location-tagged units as a GeoJSON FeatureCollection."""
    unit_list = [units] if isinstance(units, KnowledgeUnit) else list(units)
    features = []
    skipped = 0
    for unit in unit_list:
        coords = _coordinates(unit.metadata)
        if coords is None:
            skipped += 1
            continue
        features.append(_feature(unit, coords))

    collection: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if include_summary:
        collection["metadata"] = {
            "units_scanned": len(unit_list),
            "features_exported": len(features),
            "units_without_coordinates": skipped,
        }
    text = json.dumps(collection, ensure_ascii=False, sort_keys=True, indent=2)
    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    return text


def _feature(unit: KnowledgeUnit, coords: tuple[float, float]) -> dict[str, Any]:
    lon, lat = coords
    return {
        "type": "Feature",
        "id": unit.id or unit.source_id,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "id": unit.id,
            "source_id": unit.source_id,
            "title": unit.title,
            "content_type": _json_value(unit.content_type),
            "source_project": _json_value(unit.source_project),
            "tags": _json_value(unit.tags),
            "created_at": _json_value(unit.created_at),
            "updated_at": _json_value(unit.updated_at),
            "metadata": _metadata_properties(unit.metadata),
        },
    }


def _coordinates(metadata: Mapping[str, Any]) -> tuple[float, float] | None:
    for lat_key, lon_key in _COORDINATE_KEYS:
        lat = _float_value(_nested_value(metadata, lat_key))
        lon = _float_value(_nested_value(metadata, lon_key))
        if lat is None or lon is None:
            continue
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lon, lat)
    return None


def _metadata_properties(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return _remove_coordinate_metadata(metadata)


def _remove_coordinate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    top_level = {"lat", "lon", "lng", "latitude", "longitude"}
    nested = {
        "geo": {"lat", "lon", "latitude", "longitude"},
        "location": {"latitude", "longitude"},
    }
    cleaned: dict[str, Any] = {}
    for key, value in sorted(metadata.items(), key=lambda item: str(item[0])):
        text_key = str(key)
        if text_key in top_level:
            continue
        if text_key in nested and isinstance(value, Mapping):
            nested_value = {
                str(nested_key): _json_value(nested_item)
                for nested_key, nested_item in sorted(value.items(), key=lambda item: str(item[0]))
                if str(nested_key) not in nested[text_key]
            }
            if nested_value:
                cleaned[text_key] = nested_value
            continue
        cleaned[text_key] = _json_value(value)
    return cleaned


def _nested_value(metadata: Mapping[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata.get(key)
    current: Any = metadata
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current.get(part)
    return current


def _float_value(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return str(value)
