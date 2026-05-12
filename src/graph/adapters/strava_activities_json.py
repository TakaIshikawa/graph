"""Adapter for Strava activity JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime, parse_float, parse_int
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class StravaActivitiesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "strava_activities_json"

    @property
    def entity_types(self) -> list[str]:
        return ["activity"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types and "activity" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for record in records:
                unit = self._unit(record, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("activities"), list):
            return [item for item in parsed["activities"] if isinstance(item, dict)]
        return [parsed] if isinstance(parsed, dict) else []

    def _unit(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        activity_id = self._text(record.get("id") or record.get("activity_id"))
        name = self._text(record.get("name") or record.get("title"))
        sport = self._text(record.get("sport_type") or record.get("type") or record.get("activity_type"))
        if not activity_id and not name:
            return None
        started = parse_datetime(record.get("start_date") or record.get("start_date_local") or record.get("date"))
        updated = parse_datetime(record.get("updated_at")) or started
        metadata = {
            "activity_id": activity_id,
            "name": name,
            "sport_type": sport,
            "start_date": started.isoformat() if started else self._text(record.get("start_date")),
            "elapsed_time": parse_int(record.get("elapsed_time")),
            "moving_time": parse_int(record.get("moving_time")),
            "distance": parse_float(record.get("distance")),
            "total_elevation_gain": parse_float(record.get("total_elevation_gain")),
            "average_speed": parse_float(record.get("average_speed")),
            "kudos_count": parse_int(record.get("kudos_count")),
            "description": self._text(record.get("description")),
            "url": self._text(record.get("external_url") or record.get("url") or record.get("activity_url")),
            "source_file": source_file,
        }
        now = datetime.now(timezone.utc)
        title = name or f"Strava activity {activity_id}"
        return KnowledgeUnit(
            source_project=SourceProject.STRAVA_ACTIVITIES_JSON,
            source_id=f"strava_activities_json:{activity_id}" if activity_id else digest_source_id("strava_activities_json", title, started),
            source_entity_type="activity",
            title=title,
            content=self._content(title, metadata),
            content_type=ContentType.METADATA,
            metadata=clean_metadata(metadata),
            tags=list(dict.fromkeys(tag for tag in ["strava", "activity", sport] if tag)),
            created_at=started or now,
            updated_at=updated or started or now,
        )

    def _content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (("sport_type", "Sport"), ("distance", "Distance"), ("moving_time", "Moving time"), ("url", "URL")):
            if metadata.get(key) not in ("", None):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
