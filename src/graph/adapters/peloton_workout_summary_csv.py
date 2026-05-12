"""Adapter for compact Peloton workout summary CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_duration_seconds, parse_float, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class PelotonWorkoutSummaryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "peloton_workout_summary_csv"

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
                if sync_at and unit.created_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        workout_id = first(row, "Workout ID", "ID")
        start_time = parse_datetime(first(row, "Start Time", "Started At", "Workout Timestamp", "Date"))
        class_title = first(row, "Class Title", "Title", "Workout Title")
        instructor = first(row, "Instructor", "Instructor Name")
        discipline = first(row, "Discipline", "Fitness Discipline", "Workout Type")
        class_url = first(row, "Class URL", "Workout URL", "URL")
        if not any([workout_id, start_time, class_title, instructor, discipline, class_url]):
            return None

        metadata = {
            "workout_id": workout_id,
            "class_title": class_title,
            "instructor": instructor,
            "discipline": discipline,
            "duration_seconds": parse_duration_seconds(first(row, "Duration", "Length")),
            "output": parse_float(first(row, "Output", "Total Output")),
            "distance": parse_float(first(row, "Distance")),
            "calories": parse_int(first(row, "Calories", "Calories Burned")),
            "leaderboard_rank": parse_int(first(row, "Leaderboard Rank", "Rank")),
            "start_time": start_time.isoformat() if start_time else first(row, "Start Time", "Started At", "Workout Timestamp", "Date"),
            "class_url": class_url,
            "source_file": source_file,
        }
        now = datetime.now(timezone.utc)
        title = class_title or discipline or "Peloton workout"
        return KnowledgeUnit(
            source_project="peloton_workout_summary_csv",
            source_id=self._source_id(workout_id, start_time, title),
            source_entity_type="workout",
            title=title,
            content=self._content(title, metadata),
            content_type=ContentType.METADATA,
            metadata=clean_metadata(metadata),
            tags=list(dict.fromkeys(tag for tag in ["peloton", "workout", discipline, instructor] if tag)),
            created_at=start_time or now,
            updated_at=start_time or now,
        )

    def _source_id(self, workout_id: str, start_time: datetime | None, class_title: str) -> str:
        if workout_id:
            return digest_source_id("peloton_workout_summary_csv", workout_id)
        return digest_source_id("peloton_workout_summary_csv", start_time.isoformat() if start_time else "", class_title)

    def _content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (
            ("instructor", "Instructor"),
            ("discipline", "Discipline"),
            ("duration_seconds", "Duration seconds"),
            ("output", "Output"),
            ("distance", "Distance"),
            ("calories", "Calories"),
            ("leaderboard_rank", "Leaderboard rank"),
            ("start_time", "Start time"),
            ("class_url", "URL"),
        ):
            if metadata.get(key) not in ("", None, []):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
