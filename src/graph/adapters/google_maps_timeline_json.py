"""Adapter for Google Maps Timeline JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class GoogleMapsTimelineJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_maps_timeline_json"

    @property
    def entity_types(self) -> list[str]:
        return ["place_visit", "activity_segment"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types or self.entity_types)
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        items: list[KnowledgeUnit] = []
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
                items.append(unit)
        result.units.extend(sorted(items, key=lambda unit: (unit.created_at, unit.source_id)))
        result.edges.extend(self._chronological_edges(result.units))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.json") if child.is_file())

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, dict):
            if isinstance(parsed.get("timelineObjects"), list):
                parsed = parsed["timelineObjects"]
            elif isinstance(parsed.get("semanticSegments"), list):
                parsed = parsed["semanticSegments"]
        return [record for record in parsed if isinstance(record, dict)] if isinstance(parsed, list) else []

    def _unit_from_record(self, record: dict[str, Any], index: int, source_file: str) -> KnowledgeUnit | None:
        if isinstance(record.get("placeVisit"), dict):
            return self._place_visit(record["placeVisit"], index, source_file)
        if isinstance(record.get("activitySegment"), dict):
            return self._activity_segment(record["activitySegment"], index, source_file)
        if record.get("visit") or record.get("startTime"):
            return self._place_visit(record, index, source_file)
        return None

    def _place_visit(self, visit: dict[str, Any], index: int, source_file: str) -> KnowledgeUnit | None:
        start, end = self._time_range(visit)
        location = visit.get("location") if isinstance(visit.get("location"), dict) else {}
        name = self._text(location.get("name") or visit.get("name") or "Place visit")
        address = self._text(location.get("address"))
        coords = self._coords(location)
        metadata = {
            "start_at": start.isoformat() if start else "",
            "end_at": end.isoformat() if end else "",
            "place_name": name,
            "address": address,
            "latitude": coords[0],
            "longitude": coords[1],
            "confidence": self._text(visit.get("placeConfidence") or visit.get("confidence")),
            "source_file": source_file,
            "record": visit,
        }
        return self._unit("place_visit", name, self._content("Place", name, metadata), metadata, start, end, index)

    def _activity_segment(self, segment: dict[str, Any], index: int, source_file: str) -> KnowledgeUnit | None:
        start, end = self._time_range(segment)
        activity_type = self._activity_type(segment)
        start_coords = self._coords(segment.get("startLocation") if isinstance(segment.get("startLocation"), dict) else {})
        end_coords = self._coords(segment.get("endLocation") if isinstance(segment.get("endLocation"), dict) else {})
        metadata = {
            "start_at": start.isoformat() if start else "",
            "end_at": end.isoformat() if end else "",
            "activity_type": activity_type,
            "distance_meters": self._parse_float(segment.get("distance")),
            "confidence": self._text(segment.get("confidence")),
            "start_latitude": start_coords[0],
            "start_longitude": start_coords[1],
            "end_latitude": end_coords[0],
            "end_longitude": end_coords[1],
            "source_file": source_file,
            "record": segment,
        }
        return self._unit("activity_segment", activity_type or "Activity segment", self._content("Activity", activity_type, metadata), metadata, start, end, index)

    def _unit(self, entity_type: str, title: str, content: str, metadata: dict[str, Any], start: datetime | None, end: datetime | None, index: int) -> KnowledgeUnit:
        now = datetime.now(timezone.utc)
        source_id = self._source_id(entity_type, title, metadata.get("start_at", ""), metadata.get("end_at", ""), str(index))
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_MAPS_TIMELINE_JSON,
            source_id=source_id,
            source_entity_type=entity_type,
            title=title,
            content=content,
            content_type=ContentType.METADATA,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=["google_maps", "timeline", entity_type],
            created_at=start or now,
            updated_at=end or start or now,
        )

    def _chronological_edges(self, units: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for previous, current in zip(units, units[1:]):
            digest = hashlib.sha256(f"{previous.source_id}|next|{current.source_id}".encode("utf-8")).hexdigest()[:24]
            edges.append(
                KnowledgeEdge(
                    id=f"google_maps_timeline_json:{digest}",
                    from_unit_id=previous.source_id,
                    to_unit_id=current.source_id,
                    relation=EdgeRelation.REFERENCES,
                    source=EdgeSource.SOURCE,
                    metadata={"kind": "next_timeline_item"},
                )
            )
        return edges

    def _time_range(self, value: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
        duration = value.get("duration") if isinstance(value.get("duration"), dict) else {}
        return (
            self._parse_datetime(value.get("startTime") or duration.get("startTimestamp") or duration.get("startTimestampMs")),
            self._parse_datetime(value.get("endTime") or duration.get("endTimestamp") or duration.get("endTimestampMs")),
        )

    def _activity_type(self, segment: dict[str, Any]) -> str:
        if segment.get("activityType"):
            return self._text(segment.get("activityType")).lower()
        activities = segment.get("activities")
        if isinstance(activities, list) and activities:
            first = activities[0] if isinstance(activities[0], dict) else {}
            return self._text(first.get("activityType")).lower()
        return ""

    def _coords(self, location: dict[str, Any]) -> tuple[float | None, float | None]:
        lat = location.get("latitudeE7")
        lon = location.get("longitudeE7")
        if lat is not None and lon is not None:
            return self._parse_float(lat, scale=10000000), self._parse_float(lon, scale=10000000)
        return self._parse_float(location.get("latitude")), self._parse_float(location.get("longitude"))

    def _content(self, label: str, title: str, metadata: dict[str, Any]) -> str:
        parts = [f"{label}: {title}"]
        for key in ("start_at", "end_at", "address", "activity_type", "distance_meters", "confidence"):
            if metadata.get(key) not in ("", None):
                parts.append(f"{key}: {metadata[key]}")
        return "\n".join(parts)

    def _source_id(self, *parts: str) -> str:
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
        return f"google_maps_timeline_json:{digest}"

    def _parse_float(self, value: Any, scale: int = 1) -> float | None:
        try:
            return float(value) / scale
        except (TypeError, ValueError):
            return None

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

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
