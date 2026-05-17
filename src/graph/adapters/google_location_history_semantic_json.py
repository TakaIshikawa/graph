"""Adapter for Google Location History semantic JSON exports."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime, parse_float
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GoogleLocationHistorySemanticJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_location_history_semantic_json"

    @property
    def entity_types(self) -> list[str]:
        return ["place_visit", "activity_segment"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types or self.entity_types)
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit(record, path.name, index)
                if unit is None or unit.source_entity_type not in requested:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, dict):
            for key in ("timelineObjects", "semanticSegments"):
                if isinstance(parsed.get(key), list):
                    return [item for item in parsed[key] if isinstance(item, dict)]
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        if isinstance(record.get("placeVisit"), dict):
            return self._place_visit(record["placeVisit"], source_file, index)
        if isinstance(record.get("activitySegment"), dict):
            return self._activity_segment(record["activitySegment"], source_file, index)
        return None

    def _place_visit(self, visit: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        start, end = self._time_range(visit)
        location = visit.get("location") if isinstance(visit.get("location"), dict) else {}
        latitude, longitude = self._coords(location)
        place_name = self._text(location.get("name") or visit.get("name"))
        address = self._text(location.get("address"))
        place_id = self._text(location.get("placeId") or location.get("googleMapsPlaceId") or visit.get("placeId"))
        confidence = self._text(visit.get("placeConfidence") or visit.get("confidence"))

        if not any([start, end, place_name, address, place_id, latitude is not None, longitude is not None, confidence]):
            return None

        metadata = clean_metadata(
            {
                "place_name": place_name,
                "address": address,
                "place_id": place_id,
                "start_at": start.isoformat() if start else "",
                "end_at": end.isoformat() if end else "",
                "latitude": latitude,
                "longitude": longitude,
                "confidence": confidence,
                "source_file": source_file,
                "record": visit,
            }
        )
        title = place_name or address or "Place visit"
        return self._knowledge_unit("place_visit", title, self._place_content(metadata), metadata, start, end, index)

    def _activity_segment(self, segment: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        start, end = self._time_range(segment)
        activity_type = self._activity_type(segment)
        start_latitude, start_longitude = self._coords(segment.get("startLocation") if isinstance(segment.get("startLocation"), dict) else {})
        end_latitude, end_longitude = self._coords(segment.get("endLocation") if isinstance(segment.get("endLocation"), dict) else {})
        distance_meters = parse_float(segment.get("distance"))
        confidence = self._text(segment.get("confidence"))

        if not any([start, end, activity_type, distance_meters is not None, start_latitude is not None, start_longitude is not None, end_latitude is not None, end_longitude is not None, confidence]):
            return None

        metadata = clean_metadata(
            {
                "activity_type": activity_type,
                "start_at": start.isoformat() if start else "",
                "end_at": end.isoformat() if end else "",
                "distance_meters": distance_meters,
                "start_latitude": start_latitude,
                "start_longitude": start_longitude,
                "end_latitude": end_latitude,
                "end_longitude": end_longitude,
                "confidence": confidence,
                "source_file": source_file,
                "record": segment,
            }
        )
        title = activity_type.replace("_", " ").title() if activity_type else "Activity segment"
        return self._knowledge_unit("activity_segment", title, self._activity_content(metadata), metadata, start, end, index)

    def _knowledge_unit(
        self,
        entity_type: str,
        title: str,
        content: str,
        metadata: dict[str, Any],
        start: datetime | None,
        end: datetime | None,
        index: int,
    ) -> KnowledgeUnit:
        now = datetime.now(timezone.utc)
        timestamp = start or end or now
        source_id = digest_source_id(
            "google_location_history_semantic_json",
            entity_type,
            title,
            metadata.get("start_at"),
            metadata.get("end_at"),
            metadata.get("place_id"),
            metadata.get("distance_meters"),
            index,
        )
        return KnowledgeUnit(
            source_project="google_location_history_semantic_json",
            source_id=source_id,
            source_entity_type=entity_type,
            title=title,
            content=content,
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["google", "location-history", entity_type],
            created_at=timestamp,
            updated_at=end or start or now,
        )

    def _time_range(self, value: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
        duration = value.get("duration") if isinstance(value.get("duration"), dict) else {}
        return (
            self._parse_timestamp(value.get("startTime") or duration.get("startTimestamp") or duration.get("startTimestampMs")),
            self._parse_timestamp(value.get("endTime") or duration.get("endTimestamp") or duration.get("endTimestampMs")),
        )

    def _parse_timestamp(self, value: Any) -> datetime | None:
        text = self._text(value)
        if not text:
            return None
        if text.isdigit() and len(text) >= 12:
            return datetime.fromtimestamp(int(text) / 1000, timezone.utc)
        return parse_datetime(text)

    def _coords(self, location: dict[str, Any]) -> tuple[float | None, float | None]:
        latitude_e7 = location.get("latitudeE7")
        longitude_e7 = location.get("longitudeE7")
        if latitude_e7 is not None and longitude_e7 is not None:
            return parse_float(latitude_e7) / 10000000 if parse_float(latitude_e7) is not None else None, parse_float(longitude_e7) / 10000000 if parse_float(longitude_e7) is not None else None

        latitude = parse_float(location.get("latitude"))
        longitude = parse_float(location.get("longitude"))
        if latitude is not None or longitude is not None:
            return latitude, longitude

        lat_lng = self._text(location.get("latLng"))
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", lat_lng)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None, None

    def _activity_type(self, segment: dict[str, Any]) -> str:
        activity_type = self._text(segment.get("activityType")).lower()
        if activity_type:
            return activity_type
        activities = segment.get("activities")
        if isinstance(activities, list) and activities:
            first_activity = activities[0] if isinstance(activities[0], dict) else {}
            return self._text(first_activity.get("activityType")).lower()
        return ""

    def _place_content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Place: {metadata.get('place_name')}" if metadata.get("place_name") else "",
            f"Address: {metadata.get('address')}" if metadata.get("address") else "",
            f"Start: {metadata.get('start_at')}" if metadata.get("start_at") else "",
            f"End: {metadata.get('end_at')}" if metadata.get("end_at") else "",
            f"Confidence: {metadata.get('confidence')}" if metadata.get("confidence") else "",
        ]
        return "\n".join(part for part in parts if part)

    def _activity_content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Activity: {metadata.get('activity_type')}" if metadata.get("activity_type") else "",
            f"Start: {metadata.get('start_at')}" if metadata.get("start_at") else "",
            f"End: {metadata.get('end_at')}" if metadata.get("end_at") else "",
            f"Distance: {metadata.get('distance_meters')}" if metadata.get("distance_meters") is not None else "",
            f"Confidence: {metadata.get('confidence')}" if metadata.get("confidence") else "",
        ]
        return "\n".join(part for part in parts if part)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
