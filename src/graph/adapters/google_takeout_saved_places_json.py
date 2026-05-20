"""Adapter for Google Takeout saved places JSON exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GoogleTakeoutSavedPlacesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_takeout_saved_places_json"

    @property
    def entity_types(self) -> list[str]:
        return ["saved_place"]

    def __init__(self, path: str | Path = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "saved_place" not in set(entity_types or self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit_from_record(record, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if root.is_dir():
            return sorted(child for child in root.rglob("*.json") if child.is_file())
        return []

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, dict) and isinstance(parsed.get("features"), list):
            return [feature for feature in parsed["features"] if isinstance(feature, dict)]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if not isinstance(parsed, dict):
            return []
        for key in ("places", "saved_places", "savedPlaces", "locations", "items"):
            items = parsed.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return [parsed] if self._has_identity(parsed) else []

    def _unit_from_record(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        place = self._place_payload(record)
        name = self._first(place, "title", "name", "place_name", "placeName")
        address = self._first(place, "address", "formatted_address", "formattedAddress")
        url = self._first(place, "url", "maps_url", "mapsUrl", "google_maps_url", "googleMapsUrl")
        place_id = self._first(place, "place_id", "placeId", "google_place_id", "googlePlaceId")
        list_name = self._first(place, "list_name", "listName", "list", "label")
        notes = self._first(place, "notes", "note", "description", "comment")
        saved_at_text = self._first(place, "saved_at", "savedAt", "created_at", "createdAt", "date_saved", "dateSaved")
        saved_at = self._parse_datetime(saved_at_text)
        latitude, longitude = self._coords(record, place)

        if not name and not address and not url and not place_id:
            return None

        now = datetime.now(timezone.utc)
        metadata = {
            "name": name,
            "address": address,
            "url": url,
            "place_id": place_id,
            "list_name": list_name,
            "notes": notes,
            "latitude": latitude,
            "longitude": longitude,
            "saved_at": saved_at.isoformat() if saved_at else saved_at_text,
            "source_file": source_file,
            "record_index": index,
        }
        return KnowledgeUnit(
            source_project="google_takeout_saved_places_json",
            source_id=self._source_id(place_id, url, name, address, latitude, longitude, index),
            source_entity_type="saved_place",
            title=name or address or url or place_id,
            content=self._content(name, address, url, place_id, list_name, notes, latitude, longitude, saved_at),
            content_type=ContentType.METADATA,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=self._dedupe(["google", "saved-place", self._normalize_tag(list_name)]),
            created_at=saved_at or now,
            updated_at=saved_at or now,
        )

    def _place_payload(self, record: dict[str, Any]) -> dict[str, Any]:
        properties = record.get("properties")
        if isinstance(properties, dict):
            return properties
        location = record.get("location")
        if isinstance(location, dict):
            merged = dict(record)
            merged.update(location)
            return merged
        place = record.get("place")
        if isinstance(place, dict):
            merged = dict(record)
            merged.update(place)
            return merged
        return record

    def _coords(self, record: dict[str, Any], place: dict[str, Any]) -> tuple[float | None, float | None]:
        geometry = record.get("geometry") if isinstance(record.get("geometry"), dict) else place.get("geometry")
        if isinstance(geometry, dict):
            coordinates = geometry.get("coordinates")
            if isinstance(coordinates, list) and len(coordinates) >= 2:
                return self._parse_float(coordinates[1]), self._parse_float(coordinates[0])
            location = geometry.get("location")
            if isinstance(location, dict):
                lat, lon = self._lat_lon(location)
                if lat is not None or lon is not None:
                    return lat, lon
        coordinates = place.get("coordinates") or place.get("coordinate")
        if isinstance(coordinates, dict):
            lat, lon = self._lat_lon(coordinates)
            if lat is not None or lon is not None:
                return lat, lon
        if isinstance(coordinates, list) and len(coordinates) >= 2:
            return self._parse_float(coordinates[0]), self._parse_float(coordinates[1])
        return self._lat_lon(place)

    def _lat_lon(self, value: dict[str, Any]) -> tuple[float | None, float | None]:
        lat = self._parse_float(value.get("latitude") or value.get("lat"))
        lon = self._parse_float(value.get("longitude") or value.get("lng") or value.get("lon"))
        return lat, lon

    def _content(
        self,
        name: str,
        address: str,
        url: str,
        place_id: str,
        list_name: str,
        notes: str,
        latitude: float | None,
        longitude: float | None,
        saved_at: datetime | None,
    ) -> str:
        parts = []
        if name:
            parts.append(f"Place: {name}")
        if address:
            parts.append(f"Address: {address}")
        if list_name:
            parts.append(f"List: {list_name}")
        if notes:
            parts.append(f"Notes: {notes}")
        if place_id:
            parts.append(f"Place ID: {place_id}")
        if latitude is not None and longitude is not None:
            parts.append(f"Coordinates: {latitude}, {longitude}")
        if saved_at:
            parts.append(f"Saved: {saved_at.isoformat()}")
        if url:
            parts.append(f"URL: {url}")
        return "\n".join(parts)

    def _has_identity(self, record: dict[str, Any]) -> bool:
        return bool(self._first(record, "title", "name", "address", "url", "place_id", "placeId"))

    def _source_id(self, place_id: str, url: str, name: str, address: str, latitude: float | None, longitude: float | None, index: int) -> str:
        raw = place_id or url or "|".join([self._stable_text(name), self._stable_text(address), str(latitude or ""), str(longitude or "")]) or str(index)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"google_takeout_saved_places_json:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            text = self._text(value)
            if text:
                return text
        return ""

    def _parse_datetime(self, value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None

    def _parse_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _normalize_tag(self, value: str) -> str:
        return re.sub(r"\s+", "-", value.strip().casefold())

    def _stable_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().casefold())

    def _text(self, value: Any) -> str:
        if value is None or isinstance(value, (dict, list)):
            return ""
        return str(value).strip()

    def _dedupe(self, values: Any) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))
