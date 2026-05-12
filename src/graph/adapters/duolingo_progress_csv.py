"""Adapter for Duolingo progress CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class DuolingoProgressCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "duolingo_progress_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["lesson", "practice"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or self.entity_types)
        if not allowed.intersection(self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path.name, index)
                if unit is None or unit.source_entity_type not in allowed:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        completed_at = self._completed_at(row)
        language = first(row, "Language", "Course", "Learning Language", "Target Language", "language", "course")
        if completed_at is None or not language:
            return None

        lesson_type = first(row, "Lesson Type", "Type", "Activity Type", "Kind", "lesson_type")
        entity_type = "practice" if "practice" in lesson_type.casefold() else "lesson"
        skill = first(row, "Skill", "Unit", "Lesson", "Topic", "skill", "unit")
        course = first(row, "Course", "Path", "course")
        xp = parse_int(first(row, "XP", "Experience", "xp"))
        score = parse_float(first(row, "Score", "Accuracy", "score"))
        crowns = parse_int(first(row, "Crowns", "Crown Level", "crowns"))
        streak_day = parse_int(first(row, "Streak Day", "Streak", "streak_day"))
        mistakes = parse_int(first(row, "Mistakes", "Errors", "mistakes"))

        metadata = clean_metadata(
            {
                "completed_at": completed_at.isoformat(),
                "language": language,
                "course": course,
                "skill": skill,
                "lesson_type": lesson_type,
                "xp": xp,
                "crowns": crowns,
                "score": score,
                "streak_day": streak_day,
                "mistakes": mistakes,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.DUOLINGO_PROGRESS_CSV,
            source_id=self._source_id(row, completed_at, language, skill, lesson_type, index),
            source_entity_type=entity_type,
            title=self._title(entity_type, language, skill, lesson_type),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=self._dedupe(["duolingo", language.casefold(), entity_type]),
            created_at=completed_at,
            updated_at=completed_at,
        )

    def _completed_at(self, row: dict[str, Any]) -> datetime | None:
        combined = " ".join(part for part in [first(row, "Date", "date"), first(row, "Time", "time")] if part)
        return parse_datetime(first(row, "Timestamp", "Completed At", "Completed", "datetime", "DateTime") or combined)

    def _source_id(self, row: dict[str, Any], completed_at: datetime, language: str, skill: str, lesson_type: str, index: int) -> str:
        explicit = first(row, "ID", "Lesson ID", "Session ID", "id", "lesson_id", "session_id")
        stable = explicit or "|".join([completed_at.isoformat(), language, skill, lesson_type, str(index)])
        return digest_source_id("duolingo_progress_csv", stable)

    def _title(self, entity_type: str, language: str, skill: str, lesson_type: str) -> str:
        label = "Duolingo practice" if entity_type == "practice" else "Duolingo lesson"
        detail = skill or lesson_type or language
        return f"{label}: {detail}" if detail else label

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [f"Language: {metadata['language']}", f"Completed at: {metadata['completed_at']}"]
        for key, label in (("course", "Course"), ("skill", "Skill"), ("lesson_type", "Type"), ("xp", "XP"), ("score", "Score"), ("crowns", "Crowns"), ("streak_day", "Streak day"), ("mistakes", "Mistakes")):
            if key in metadata:
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = value.strip()
            if text and text not in result:
                result.append(text)
        return result
