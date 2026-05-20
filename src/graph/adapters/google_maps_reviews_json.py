"""Adapter for Google Maps review JSON exports."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime, parse_float, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GoogleMapsReviewsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_maps_reviews_json"

    @property
    def entity_types(self) -> list[str]:
        return ["place_review"]

    def __init__(self, path: str | Path = "") -> None:
        self.path = str(path)

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "place_review" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".json"}):
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

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._records_from_payload(parsed)

    def _records_from_payload(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("reviews", "places", "items", "data"):
            items = payload.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return [payload] if self._has_review_identity(payload) else []

    def _unit_from_record(self, record: dict[str, Any], source_file: str, record_index: int) -> KnowledgeUnit | None:
        place = self._place_payload(record)
        review = self._review_payload(record)
        place_name = self._first(record, place, "place_name", "placeName", "name", "title", "business_name", "businessName")
        place_id = self._first(record, place, "place_id", "placeId", "google_place_id", "googlePlaceId", "cid")
        address = self._first(record, place, "address", "formatted_address", "formattedAddress")
        url = self._first(record, place, review, "url", "maps_url", "mapsUrl", "google_maps_url", "googleMapsUrl", "place_url", "placeUrl")
        review_id = self._first(record, review, "review_id", "reviewId", "id")
        review_text = self._first(record, review, "review_text", "reviewText", "text", "comment", "description", "content")
        categories = split_values(self._first(record, place, "categories", "category", "types", "tags"))
        rating = parse_float(self._first(record, review, "rating", "stars", "score"))
        reviewed_text = self._first(record, review, "reviewed_at", "reviewedAt", "created_at", "createdAt", "date", "time")
        visited_text = self._first(record, review, "visited_at", "visitedAt", "visit_date", "visitDate", "visited")
        updated_text = self._first(record, review, "updated_at", "updatedAt", "modified_at", "modifiedAt")
        reviewed_at = parse_datetime(reviewed_text)
        visited_at = parse_datetime(visited_text)
        updated_at = parse_datetime(updated_text)
        latitude, longitude = self._coords(record, place)

        if not any([place_name, place_id, address, url, review_text, rating is not None]):
            return None

        observed_at = updated_at or reviewed_at or visited_at
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "place_name": place_name,
                "place_id": place_id,
                "address": address,
                "latitude": latitude,
                "longitude": longitude,
                "rating": rating,
                "review_text": review_text,
                "categories": categories,
                "url": url,
                "review_id": review_id,
                "reviewed_at": reviewed_at.isoformat() if reviewed_at else reviewed_text,
                "visited_at": visited_at.isoformat() if visited_at else visited_text,
                "updated_at": updated_at.isoformat() if updated_at else updated_text,
                "source_file": source_file,
                "record_index": record_index,
                "record": record,
            }
        )
        title = place_name or address or url or "Google Maps review"
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_MAPS_REVIEWS_JSON,
            source_id=self._source_id(review_id, place_id, place_name, rating, observed_at),
            source_entity_type="place_review",
            title=title,
            content=self._content(metadata),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=self._dedupe(["google_maps", "review", *categories]),
            created_at=reviewed_at or visited_at or observed_at or now,
            updated_at=observed_at or now,
        )

    def _place_payload(self, record: dict[str, Any]) -> dict[str, Any]:
        for key in ("place", "location", "business", "venue"):
            value = record.get(key)
            if isinstance(value, dict):
                merged = dict(record)
                merged.update(value)
                return merged
        return record

    def _review_payload(self, record: dict[str, Any]) -> dict[str, Any]:
        value = record.get("review")
        if isinstance(value, dict):
            merged = dict(record)
            merged.update(value)
            return merged
        return record

    def _coords(self, record: dict[str, Any], place: dict[str, Any]) -> tuple[float | None, float | None]:
        for source in (place, record):
            lat = parse_float(source.get("latitude") or source.get("lat"))
            lon = parse_float(source.get("longitude") or source.get("lng") or source.get("lon"))
            if lat is not None or lon is not None:
                return lat, lon
            coords = source.get("coordinates") or source.get("coordinate")
            if isinstance(coords, dict):
                lat = parse_float(coords.get("latitude") or coords.get("lat"))
                lon = parse_float(coords.get("longitude") or coords.get("lng") or coords.get("lon"))
                if lat is not None or lon is not None:
                    return lat, lon
            if isinstance(coords, list) and len(coords) >= 2:
                return parse_float(coords[0]), parse_float(coords[1])
            geometry = source.get("geometry")
            if isinstance(geometry, dict):
                geo_coords = geometry.get("coordinates")
                if isinstance(geo_coords, list) and len(geo_coords) >= 2:
                    return parse_float(geo_coords[1]), parse_float(geo_coords[0])
        return None, None

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = []
        for key, label in (
            ("place_name", "Place"),
            ("rating", "Rating"),
            ("review_text", "Review"),
            ("address", "Address"),
            ("categories", "Categories"),
            ("visited_at", "Visited"),
            ("reviewed_at", "Reviewed"),
            ("updated_at", "Updated"),
            ("url", "URL"),
        ):
            if key in metadata:
                value = ", ".join(metadata[key]) if isinstance(metadata[key], list) else metadata[key]
                parts.append(f"{label}: {value}")
        if metadata.get("latitude") is not None and metadata.get("longitude") is not None:
            parts.append(f"Coordinates: {metadata['latitude']}, {metadata['longitude']}")
        return "\n".join(str(part) for part in parts if part)

    def _source_id(self, review_id: str, place_id: str, place_name: str, rating: float | None, observed_at: datetime | None) -> str:
        date = observed_at.isoformat() if observed_at else ""
        if review_id:
            return digest_source_id("google_maps_reviews_json", review_id)
        if place_id:
            return digest_source_id("google_maps_reviews_json", place_id, date)
        return digest_source_id("google_maps_reviews_json", self._stable_text(place_name), rating, date)

    def _has_review_identity(self, record: dict[str, Any]) -> bool:
        return bool(self._first(record, "name", "title", "place_name", "review_text", "rating", "url"))

    def _first(self, *args: Any) -> str:
        *rows, keys = args
        if not isinstance(keys, tuple):
            keys = tuple(str(key) for key in args if isinstance(key, str))
            rows = [arg for arg in args if isinstance(arg, dict)]
        for row in rows:
            if not isinstance(row, dict):
                continue
            compact = {self._normalize_key(key): value for key, value in row.items()}
            for key in keys:
                value = row.get(key)
                if value is None:
                    value = compact.get(self._normalize_key(key))
                text = self._text(value)
                if text:
                    return text
        return ""

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _stable_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().casefold())

    def _text(self, value: Any) -> str:
        if value is None or isinstance(value, dict):
            return ""
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        return str(value).strip()

    def _dedupe(self, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))
