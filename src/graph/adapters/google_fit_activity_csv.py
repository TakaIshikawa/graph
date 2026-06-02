"""Adapter for Google Fit activity CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_duration_seconds, parse_float, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GoogleFitActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_fit_activity_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["activity"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "activity" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows, start=2):
                unit = self._unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, source_row: int) -> KnowledgeUnit | None:
        start = parse_datetime(first(row, "Start Time", "Start", "Start Date", "Begin Time", "From"))
        end = parse_datetime(first(row, "End Time", "End", "End Date", "Finish Time", "To"))
        activity = first(row, "Activity", "Activity Type", "Type", "Move Minutes Activity", "Exercise")
        steps = parse_int(first(row, "Steps", "Step Count", "Step count"))
        distance = parse_float(first(row, "Distance", "Distance (m)", "Distance (km)", "Distance (mi)"))
        calories = parse_float(first(row, "Calories", "Calories (kcal)", "Calories Burned", "Energy"))
        duration = parse_duration_seconds(first(row, "Duration", "Duration (s)", "Duration Seconds", "Elapsed Time"))
        if not any([start, end, activity, steps is not None, distance is not None, calories is not None, duration is not None]):
            return None
        now = datetime.now(timezone.utc)
        created_at = start or end or now
        updated_at = end or created_at
        metadata = clean_metadata(
            {
                "start_time": start.isoformat() if start else first(row, "Start Time", "Start"),
                "end_time": end.isoformat() if end else first(row, "End Time", "End"),
                "activity_type": activity,
                "steps": steps,
                "distance": distance,
                "distance_unit": self._distance_unit(row),
                "calories": calories,
                "duration_seconds": duration,
                "source_file": source_file,
                "source_row": source_row,
            }
        )
        title = f"Google Fit {activity or 'activity'}"
        if start:
            title = f"{title} {start.date().isoformat()}"
        return KnowledgeUnit(
            source_project="google_fit_activity_csv",
            source_id=digest_source_id("google_fit_activity_csv", start.isoformat() if start else "", activity, source_file, source_row),
            source_entity_type="activity",
            title=title,
            content=self._content(title, metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["google_fit", "activity", activity] if tag)),
            created_at=created_at,
            updated_at=updated_at,
        )

    def _distance_unit(self, row: dict[str, Any]) -> str:
        for key in row:
            lowered = str(key).casefold()
            if "distance" in lowered and "(km)" in lowered:
                return "km"
            if "distance" in lowered and "(mi)" in lowered:
                return "mi"
            if "distance" in lowered and "(m)" in lowered:
                return "m"
        return first(row, "Distance Unit", "Unit")

    def _content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (("start_time", "Start"), ("end_time", "End"), ("steps", "Steps"), ("distance", "Distance"), ("calories", "Calories"), ("duration_seconds", "Duration seconds")):
            if key in metadata:
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
