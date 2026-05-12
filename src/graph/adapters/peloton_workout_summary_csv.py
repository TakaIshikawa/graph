"""Adapter for compact Peloton workout summary CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_duration_seconds, parse_float, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class PelotonWorkoutSummaryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "peloton_workout_summary_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["workout", "workout_month"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        workout_units: list[KnowledgeUnit] = []
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
                workout_units.append(unit)

        month_units = self._month_units(workout_units)
        if "workout" in allowed_types:
            result.units.extend(workout_units)
        if "workout_month" in allowed_types:
            result.units.extend(month_units)
        if {"workout", "workout_month"}.issubset(allowed_types):
            result.edges.extend(self._month_edges(month_units, workout_units))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
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

    def _month_units(self, workouts: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for workout in workouts:
            month = workout.created_at.strftime("%Y-%m")
            grouped.setdefault(month, []).append(workout)

        units: list[KnowledgeUnit] = []
        for month, month_workouts in sorted(grouped.items()):
            durations = [value for workout in month_workouts if (value := workout.metadata.get("duration_seconds")) is not None]
            outputs = [value for workout in month_workouts if (value := workout.metadata.get("output")) is not None]
            distances = [value for workout in month_workouts if (value := workout.metadata.get("distance")) is not None]
            calories = [value for workout in month_workouts if (value := workout.metadata.get("calories")) is not None]
            metadata = {
                "month": month,
                "workout_count": len(month_workouts),
                "total_duration_seconds": sum(durations),
                "total_output": sum(outputs),
                "total_distance": sum(distances),
                "total_calories": sum(calories),
                "disciplines": sorted({str(workout.metadata.get("discipline")) for workout in month_workouts if workout.metadata.get("discipline")}),
                "instructors": sorted({str(workout.metadata.get("instructor")) for workout in month_workouts if workout.metadata.get("instructor")}),
                "workout_source_ids": sorted(workout.source_id for workout in month_workouts),
                "first_workout_at": min(workout.created_at for workout in month_workouts).isoformat(),
                "latest_workout_at": max(workout.created_at for workout in month_workouts).isoformat(),
            }
            units.append(
                KnowledgeUnit(
                    source_project="peloton_workout_summary_csv",
                    source_id=digest_source_id("peloton_workout_summary_csv_month", month),
                    source_entity_type="workout_month",
                    title=f"Peloton workouts {month}",
                    content=f"Peloton workouts {month}\nWorkouts: {len(month_workouts)}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=["peloton", "workout-month", month],
                    created_at=min(workout.created_at for workout in month_workouts),
                    updated_at=max(workout.updated_at for workout in month_workouts),
                )
            )
        return units

    def _month_edges(self, months: list[KnowledgeUnit], workouts: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        month_ids = {str(month.metadata.get("month")): month.source_id for month in months}
        edges: list[KnowledgeEdge] = []
        for workout in workouts:
            month_id = month_ids.get(workout.created_at.strftime("%Y-%m"))
            if not month_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=digest_source_id("peloton_workout_summary_csv_month_edge", month_id, workout.source_id),
                    from_unit_id=month_id,
                    to_unit_id=workout.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={"relation_type": "month_contains_workout", "month": workout.created_at.strftime("%Y-%m")},
                )
            )
        return edges

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
