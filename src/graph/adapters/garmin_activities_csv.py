"""Adapter for Garmin Connect activities CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_duration_seconds, parse_float, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GarminActivitiesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "garmin_activities_csv"

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
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        activity_id = first(row, "Activity ID", "ID")
        title = first(row, "Title", "Name")
        sport = first(row, "Activity Type", "Type")
        if not activity_id and not title and not sport:
            return None
        date = parse_datetime(first(row, "Date", "Start Time", "Start"))
        metadata = {
            "activity_id": activity_id,
            "activity_type": sport,
            "favorite": first(row, "Favorite"),
            "title": title,
            "date": date.isoformat() if date else first(row, "Date"),
            "distance": parse_float(first(row, "Distance")),
            "calories": parse_int(first(row, "Calories")),
            "duration_seconds": parse_duration_seconds(first(row, "Time", "Duration")),
            "avg_hr": parse_int(first(row, "Avg HR", "Average HR")),
            "max_hr": parse_int(first(row, "Max HR")),
            "aerobic_te": parse_float(first(row, "Aerobic TE")),
            "avg_speed": parse_float(first(row, "Avg Speed")),
            "max_speed": parse_float(first(row, "Max Speed")),
            "total_ascent": parse_float(first(row, "Total Ascent")),
            "total_descent": parse_float(first(row, "Total Descent")),
            "source_file": source_file,
        }
        now = datetime.now(timezone.utc)
        name = title or sport or f"Garmin activity {activity_id}"
        return KnowledgeUnit(
            source_project=SourceProject.GARMIN_ACTIVITIES_CSV,
            source_id=f"garmin_activities_csv:{activity_id}" if activity_id else digest_source_id("garmin_activities_csv", name, date),
            source_entity_type="activity",
            title=name,
            content=self._content(name, metadata),
            content_type=ContentType.METADATA,
            metadata=clean_metadata(metadata),
            tags=list(dict.fromkeys(tag for tag in ["garmin", "activity", sport] if tag)),
            created_at=date or now,
            updated_at=date or now,
        )

    def _content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (("activity_type", "Type"), ("distance", "Distance"), ("duration_seconds", "Duration seconds")):
            if metadata.get(key) not in ("", None):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
