"""Adapter for Peloton workout CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_duration_seconds, parse_float, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class PelotonWorkoutsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "peloton_workouts_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["workout"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types and "workout" not in entity_types:
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
        timestamp = parse_datetime(first(row, "Workout Timestamp", "Date", "Start Time"))
        discipline = first(row, "Fitness Discipline", "Discipline")
        title = first(row, "Title", "Workout Title")
        instructor = first(row, "Instructor Name", "Instructor")
        url = first(row, "Workout URL", "URL")
        if not any([timestamp, discipline, title, instructor, url]):
            return None
        metadata = {
            "workout_timestamp": timestamp.isoformat() if timestamp else first(row, "Workout Timestamp"),
            "fitness_discipline": discipline,
            "title": title,
            "instructor_name": instructor,
            "length_seconds": parse_duration_seconds(first(row, "Length", "Duration")),
            "total_output": parse_float(first(row, "Total Output")),
            "calories_burned": parse_int(first(row, "Calories Burned", "Calories")),
            "distance": parse_float(first(row, "Distance")),
            "avg_watts": parse_float(first(row, "Avg Watts")),
            "avg_resistance": parse_float(first(row, "Avg Resistance")),
            "avg_cadence": parse_float(first(row, "Avg Cadence")),
            "avg_speed": parse_float(first(row, "Avg Speed")),
            "avg_heartrate": parse_float(first(row, "Avg Heartrate", "Avg Heart Rate")),
            "workout_url": url,
            "source_file": source_file,
        }
        now = datetime.now(timezone.utc)
        name = title or discipline or "Peloton workout"
        return KnowledgeUnit(
            source_project=SourceProject.PELOTON_WORKOUTS_CSV,
            source_id=digest_source_id("peloton_workouts_csv", url, name, timestamp),
            source_entity_type="workout",
            title=name,
            content=self._content(name, metadata),
            content_type=ContentType.METADATA,
            metadata=clean_metadata(metadata),
            tags=list(dict.fromkeys(tag for tag in ["peloton", "workout", discipline, instructor] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (("fitness_discipline", "Discipline"), ("instructor_name", "Instructor"), ("workout_url", "URL")):
            if metadata.get(key):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
