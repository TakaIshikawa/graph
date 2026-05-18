"""Adapter for Google Takeout semantic location history JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GoogleLocationSemanticHistoryJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_location_semantic_history_json"

    @property
    def entity_types(self) -> list[str]:
        return ["place_visit", "activity_segment"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types or self.entity_types)
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []

        for path in self._iter_paths():
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit_from_record(record, index, path.name)
                if unit is None or unit.source_entity_type not in requested:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.created_at, unit.source_id)))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.json") if path.is_file())

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, dict):
            if isinstance(parsed.get("timelineObjects"), list):
                parsed = parsed["timelineObjects"]
            elif isinstance(parsed.get("semanticSegments"), list):
                parsed = parsed["semanticSegments"]
        if not isinstance(parsed, list):
            return []
        return [record for record in parsed if isinstance(record, dict)]

    def _unit_from_record(
        self,
        record: dict[str, Any],
        record_index: int,
        source_file: str,
    ) -> KnowledgeUnit | None:
        if isinstance(record.get("placeVisit"), dict):
            return self._place_visit(record["placeVisit"], record_index, source_file)
        if isinstance(record.get("activitySegment"), dict):
            return self._activity_segment(record["activitySegment"], record_index, source_file)
        return None

    def _place_visit(
        self,
        visit: dict[str, Any],
        record_index: int,
        source_file: str,
    ) -> KnowledgeUnit:
        start_at, end_at = self._time_range(visit)
        location = visit.get("location") if isinstance(visit.get("location"), dict) else {}
        place_name = self._text(location.get("name") or visit.get("name") or "Place visit")
        address = self._text(location.get("address") or visit.get("address"))
        confidence = self._text(visit.get("placeConfidence") or visit.get("confidence"))
        coords = self._coords(location)
        metadata = {
            "place_name": place_name,
            "address": address,
            "start_at": self._isoformat(start_at),
            "end_at": self._isoformat(end_at),
            "confidence": confidence,
            "place_id": self._text(location.get("placeId")),
            "latitude": coords[0],
            "longitude": coords[1],
            "source_file": source_file,
            "record_index": record_index,
        }
        content = self._place_content(place_name, address, start_at, end_at, confidence)
        return self._unit("place_visit", place_name, content, metadata, start_at, end_at, record_index)

    def _activity_segment(
        self,
        segment: dict[str, Any],
        record_index: int,
        source_file: str,
    ) -> KnowledgeUnit:
        start_at, end_at = self._time_range(segment)
        activity_type = self._activity_type(segment)
        distance = self._float(segment.get("distance"))
        confidence = self._text(segment.get("confidence"))
        start_coords = self._coords(segment.get("startLocation") if isinstance(segment.get("startLocation"), dict) else {})
        end_coords = self._coords(segment.get("endLocation") if isinstance(segment.get("endLocation"), dict) else {})
        metadata = {
            "activity_type": activity_type,
            "distance_meters": distance,
            "start_at": self._isoformat(start_at),
            "end_at": self._isoformat(end_at),
            "confidence": confidence,
            "start_latitude": start_coords[0],
            "start_longitude": start_coords[1],
            "end_latitude": end_coords[0],
            "end_longitude": end_coords[1],
            "source_file": source_file,
            "record_index": record_index,
        }
        title = activity_type or "Activity segment"
        content = self._activity_content(title, distance, start_at, end_at, confidence)
        return self._unit("activity_segment", title, content, metadata, start_at, end_at, record_index)

    def _unit(
        self,
        entity_type: str,
        title: str,
        content: str,
        metadata: dict[str, Any],
        start_at: datetime | None,
        end_at: datetime | None,
        record_index: int,
    ) -> KnowledgeUnit:
        now = datetime.now(timezone.utc)
        source_id = self._source_id(
            entity_type,
            title,
            metadata.get("start_at", ""),
            metadata.get("end_at", ""),
            metadata.get("address", ""),
            str(record_index),
        )
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_LOCATION_SEMANTIC_HISTORY_JSON,
            source_id=source_id,
            source_entity_type=entity_type,
            title=title,
            content=content,
            content_type=ContentType.METADATA,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=["google_location", "semantic_history", entity_type],
            created_at=start_at or now,
            updated_at=end_at or start_at or now,
        )

    def _place_content(
        self,
        place_name: str,
        address: str,
        start_at: datetime | None,
        end_at: datetime | None,
        confidence: str,
    ) -> str:
        parts = [f"Place visit: {place_name}"]
        if address:
            parts.append(f"Address: {address}")
        parts.append(f"Time range: {self._range_text(start_at, end_at)}")
        if confidence:
            parts.append(f"Confidence: {confidence}")
        return "\n".join(parts)

    def _activity_content(
        self,
        activity_type: str,
        distance: float | None,
        start_at: datetime | None,
        end_at: datetime | None,
        confidence: str,
    ) -> str:
        parts = [f"Activity segment: {activity_type}"]
        if distance is not None:
            parts.append(f"Distance: {distance:g} meters")
        parts.append(f"Time range: {self._range_text(start_at, end_at)}")
        if confidence:
            parts.append(f"Confidence: {confidence}")
        return "\n".join(parts)

    def _time_range(self, value: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
        duration = value.get("duration") if isinstance(value.get("duration"), dict) else {}
        return (
            self._parse_datetime(value.get("startTime") or duration.get("startTimestamp") or duration.get("startTimestampMs")),
            self._parse_datetime(value.get("endTime") or duration.get("endTimestamp") or duration.get("endTimestampMs")),
        )

    def _activity_type(self, segment: dict[str, Any]) -> str:
        direct = self._text(segment.get("activityType"))
        if direct:
            return direct.lower()
        activities = segment.get("activities")
        if isinstance(activities, list):
            for activity in activities:
                if not isinstance(activity, dict):
                    continue
                value = self._text(activity.get("activityType"))
                if value:
                    return value.lower()
        return ""

    def _coords(self, location: dict[str, Any]) -> tuple[float | None, float | None]:
        if location.get("latitudeE7") is not None and location.get("longitudeE7") is not None:
            return self._float(location.get("latitudeE7"), scale=10000000), self._float(location.get("longitudeE7"), scale=10000000)
        return self._float(location.get("latitude")), self._float(location.get("longitude"))

    def _parse_datetime(self, value: Any) -> datetime | None:
        text = self._text(value)
        if not text:
            return None
        if text.isdigit() and len(text) >= 12:
            return datetime.fromtimestamp(int(text) / 1000, timezone.utc)
        try:
            return self._ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _isoformat(self, value: datetime | None) -> str:
        return value.isoformat() if value else ""

    def _range_text(self, start_at: datetime | None, end_at: datetime | None) -> str:
        start_text = self._isoformat(start_at) or "unknown"
        end_text = self._isoformat(end_at) or "unknown"
        return f"{start_text} to {end_text}"

    def _source_id(self, *parts: Any) -> str:
        digest = hashlib.sha256("|".join(self._text(part) for part in parts).encode("utf-8")).hexdigest()[:24]
        return f"google_location_semantic_history_json:{digest}"

    def _float(self, value: Any, scale: int = 1) -> float | None:
        try:
            return float(value) / scale
        except (TypeError, ValueError):
            return None

    def _text(self, value: Any) -> str:
        if value is None or isinstance(value, (dict, list)):
            return ""
        return str(value).strip()
