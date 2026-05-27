"""Adapter for Duolingo learning session CSV exports."""

from __future__ import annotations

import csv
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class DuolingoLearningCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "duolingo_learning_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["learning_session"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "learning_session" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        learned_at = parse_datetime(first(row, "Date", "Timestamp", "Completed At", "Completed", "datetime"))
        course = first(row, "Course", "Language", "Learning Language", "Target Language")
        skill = first(row, "Skill", "Unit", "Topic")
        lesson = first(row, "Lesson", "Lesson Name", "Activity", "Exercise")
        xp = parse_int(first(row, "XP", "Experience", "XP Earned"))
        if learned_at is None or not any((course, skill, lesson, xp is not None)):
            return None
        correct = parse_int(first(row, "Correct", "Correct Count", "Answers Correct"))
        mistakes = parse_int(first(row, "Mistakes", "Errors", "Incorrect"))
        time_spent = first(row, "Time Spent", "Duration", "Elapsed")
        notes = first(row, "Notes", "Note")
        metadata = clean_metadata(
            {
                "date": learned_at.isoformat(),
                "course": course,
                "skill": skill,
                "lesson": lesson,
                "xp": xp,
                "correct": correct,
                "mistakes": mistakes,
                "time_spent": time_spent,
                "notes": notes,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="duolingo_learning_csv",
            source_id=digest_source_id("duolingo_learning_csv", first(row, "ID", "Session ID") or learned_at.isoformat(), course, skill, lesson, xp, index),
            source_entity_type="learning_session",
            title=self._title(course, skill, lesson),
            content=self._content(course, skill, lesson, xp, correct, mistakes, time_spent, notes),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=[tag for tag in ["duolingo", course.casefold() if course else ""] if tag],
            created_at=learned_at,
            updated_at=learned_at,
        )

    def _title(self, course: str, skill: str, lesson: str) -> str:
        subject = " - ".join(part for part in (course, skill, lesson) if part)
        return f"Duolingo learning session: {subject}" if subject else "Duolingo learning session"

    def _content(self, course: str, skill: str, lesson: str, xp: int | None, correct: int | None, mistakes: int | None, time_spent: str, notes: str) -> str:
        parts = [self._title(course, skill, lesson)]
        if xp is not None:
            parts.append(f"XP: {xp}")
        if correct is not None:
            parts.append(f"Correct: {correct}")
        if mistakes is not None:
            parts.append(f"Mistakes: {mistakes}")
        if time_spent:
            parts.append(f"Time spent: {time_spent}")
        if notes:
            parts.append(notes)
        return "\n".join(parts)
