"""Adapter for Duolingo progress CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class DuolingoProgressCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "duolingo_progress_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["course", "skill", "lesson", "practice", "language"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else {"lesson", "practice"}
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
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        activities = sorted(result.units, key=lambda unit: (unit.updated_at, unit.source_id))
        needs_courses = "course" in allowed or "language" in allowed
        needs_skills = "skill" in allowed or "language" in allowed
        courses = self._course_units(activities) if needs_courses else []
        skills = self._skill_units(activities) if needs_skills else []
        languages = self._language_units(activities, courses, skills) if "language" in allowed else []
        result.units = []
        if "language" in allowed:
            result.units.extend(languages)
        if "course" in allowed:
            result.units.extend(courses)
        if "skill" in allowed:
            result.units.extend(skills)
        for entity_type in ("lesson", "practice"):
            if entity_type in allowed:
                result.units.extend(unit for unit in activities if unit.source_entity_type == entity_type)
        if {"course", "skill"}.issubset(allowed):
            result.edges.extend(self._course_skill_edges(courses, skills))
        if {"language", "course"}.issubset(allowed):
            result.edges.extend(self._language_course_edges(languages, courses))
        if {"language", "skill"}.issubset(allowed) and "course" not in allowed:
            result.edges.extend(self._language_skill_edges(languages, skills))
        if "skill" in allowed and {"lesson", "practice"}.intersection(allowed):
            result.edges.extend(self._skill_activity_edges(skills, activities, allowed))
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
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

    def _course_units(self, activities: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[tuple[str, str], list[KnowledgeUnit]] = {}
        for activity in activities:
            grouped.setdefault(self._course_identity(activity.metadata), []).append(activity)
        units: list[KnowledgeUnit] = []
        for identity, course_activities in grouped.items():
            first_unit = course_activities[0]
            title = str(first_unit.metadata.get("course") or first_unit.metadata.get("language") or identity[1])
            created_at = min(unit.created_at for unit in course_activities)
            updated_at = max(unit.updated_at for unit in course_activities)
            metadata = self._aggregate_metadata(course_activities)
            metadata.update(
                {
                    "course": str(first_unit.metadata.get("course") or ""),
                    "language": str(first_unit.metadata.get("language") or ""),
                    "skill_source_ids": [],
                    "activity_source_ids": [unit.source_id for unit in course_activities],
                }
            )
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.DUOLINGO_PROGRESS_CSV,
                    source_id=self._course_source_id(identity),
                    source_entity_type="course",
                    title=f"Duolingo course: {title}",
                    content=f"Duolingo course: {title}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=self._dedupe(["duolingo", str(first_unit.metadata.get("language") or "").casefold(), "course"]),
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return units

    def _skill_units(self, activities: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[tuple[str, str, str], list[KnowledgeUnit]] = {}
        for activity in activities:
            key = self._skill_identity(activity.metadata)
            if key[2]:
                grouped.setdefault(key, []).append(activity)
        units: list[KnowledgeUnit] = []
        for identity, skill_activities in grouped.items():
            first_unit = skill_activities[0]
            skill = str(first_unit.metadata.get("skill") or identity[2])
            created_at = min(unit.created_at for unit in skill_activities)
            updated_at = max(unit.updated_at for unit in skill_activities)
            metadata = self._aggregate_metadata(skill_activities)
            metadata.update(
                {
                    "course": str(first_unit.metadata.get("course") or ""),
                    "language": str(first_unit.metadata.get("language") or ""),
                    "skill": skill,
                    "activity_source_ids": [unit.source_id for unit in skill_activities],
                }
            )
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.DUOLINGO_PROGRESS_CSV,
                    source_id=self._skill_source_id(identity),
                    source_entity_type="skill",
                    title=f"Duolingo skill: {skill}",
                    content=f"Duolingo skill: {skill}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=self._dedupe(["duolingo", str(first_unit.metadata.get("language") or "").casefold(), "skill"]),
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return units

    def _aggregate_metadata(self, activities: list[KnowledgeUnit]) -> dict[str, Any]:
        scores = [float(unit.metadata["score"]) for unit in activities if unit.metadata.get("score") is not None]
        return {
            "lesson_count": sum(1 for unit in activities if unit.source_entity_type == "lesson"),
            "practice_count": sum(1 for unit in activities if unit.source_entity_type == "practice"),
            "total_xp": sum(int(unit.metadata.get("xp") or 0) for unit in activities),
            "average_score": (sum(scores) / len(scores)) if scores else None,
            "mistake_total": sum(int(unit.metadata.get("mistakes") or 0) for unit in activities),
            "first_completed_at": min(unit.created_at for unit in activities).isoformat(),
            "last_completed_at": max(unit.updated_at for unit in activities).isoformat(),
            "source_files": sorted({str(unit.metadata.get("source_file")) for unit in activities if unit.metadata.get("source_file")}),
        }

    def _language_units(
        self,
        activities: list[KnowledgeUnit],
        courses: list[KnowledgeUnit],
        skills: list[KnowledgeUnit],
    ) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for activity in activities:
            language = str(activity.metadata.get("language") or "").strip()
            if not language:
                continue
            identity = self._language_identity(activity.metadata)
            grouped.setdefault(identity, []).append(activity)
            names.setdefault(identity, language)

        course_ids_by_language: dict[str, list[str]] = {}
        for course in courses:
            language_identity = self._language_identity(course.metadata)
            course_ids_by_language.setdefault(language_identity, []).append(course.source_id)
        skill_ids_by_language: dict[str, list[str]] = {}
        for skill in skills:
            language_identity = self._language_identity(skill.metadata)
            skill_ids_by_language.setdefault(language_identity, []).append(skill.source_id)

        units: list[KnowledgeUnit] = []
        for identity, language_activities in sorted(grouped.items()):
            language = names[identity]
            metadata = self._aggregate_metadata(language_activities)
            course_source_ids = sorted(set(course_ids_by_language.get(identity, [])))
            skill_source_ids = sorted(set(skill_ids_by_language.get(identity, [])))
            metadata.update(
                {
                    "language": language,
                    "normalized_language": identity,
                    "course_count": len(course_source_ids),
                    "skill_count": len(skill_source_ids),
                    "course_source_ids": course_source_ids,
                    "skill_source_ids": skill_source_ids,
                    "activity_source_ids": [unit.source_id for unit in language_activities],
                }
            )
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.DUOLINGO_PROGRESS_CSV,
                    source_id=self._language_source_id(identity),
                    source_entity_type="language",
                    title=f"Duolingo language: {language}",
                    content=f"Duolingo language: {language}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=self._dedupe(["duolingo", language.casefold(), "language"]),
                    created_at=min(unit.created_at for unit in language_activities),
                    updated_at=max(unit.updated_at for unit in language_activities),
                )
            )
        return units

    def _course_skill_edges(self, courses: list[KnowledgeUnit], skills: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        course_ids = {self._course_identity(course.metadata): course.source_id for course in courses}
        edges: list[KnowledgeEdge] = []
        for skill in skills:
            course_id = course_ids.get(self._course_identity(skill.metadata))
            if course_id:
                edges.append(self._edge(course_id, skill.source_id, "course_contains_skill"))
        return list({edge.id: edge for edge in edges}.values())

    def _skill_activity_edges(self, skills: list[KnowledgeUnit], activities: list[KnowledgeUnit], allowed: set[str]) -> list[KnowledgeEdge]:
        skill_ids = {self._skill_identity(skill.metadata): skill.source_id for skill in skills}
        edges: list[KnowledgeEdge] = []
        for activity in activities:
            if activity.source_entity_type not in allowed:
                continue
            skill_id = skill_ids.get(self._skill_identity(activity.metadata))
            if skill_id:
                edges.append(self._edge(skill_id, activity.source_id, "skill_contains_activity"))
        return list({edge.id: edge for edge in edges}.values())

    def _language_course_edges(self, languages: list[KnowledgeUnit], courses: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        language_ids = {self._language_identity(language.metadata): language.source_id for language in languages}
        edges: list[KnowledgeEdge] = []
        for course in courses:
            language_id = language_ids.get(self._language_identity(course.metadata))
            if language_id:
                edges.append(self._edge(language_id, course.source_id, "language_contains_course"))
        return list({edge.id: edge for edge in edges}.values())

    def _language_skill_edges(self, languages: list[KnowledgeUnit], skills: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        language_ids = {self._language_identity(language.metadata): language.source_id for language in languages}
        edges: list[KnowledgeEdge] = []
        for skill in skills:
            language_id = language_ids.get(self._language_identity(skill.metadata))
            if language_id:
                edges.append(self._edge(language_id, skill.source_id, "language_contains_skill"))
        return list({edge.id: edge for edge in edges}.values())

    def _course_identity(self, metadata: dict[str, Any]) -> tuple[str, str]:
        course = " ".join(str(metadata.get("course") or "").casefold().split())
        language = " ".join(str(metadata.get("language") or "").casefold().split())
        return course, language

    def _skill_identity(self, metadata: dict[str, Any]) -> tuple[str, str, str]:
        course, language = self._course_identity(metadata)
        skill = " ".join(str(metadata.get("skill") or "").casefold().split())
        return course, language, skill

    def _language_identity(self, metadata: dict[str, Any]) -> str:
        return " ".join(str(metadata.get("language") or "").casefold().split())

    def _course_source_id(self, identity: tuple[str, str]) -> str:
        return digest_source_id("duolingo_progress_csv:course", *identity)

    def _skill_source_id(self, identity: tuple[str, str, str]) -> str:
        return digest_source_id("duolingo_progress_csv:skill", *identity)

    def _language_source_id(self, identity: str) -> str:
        return digest_source_id("duolingo_progress_csv:language", identity)

    def _edge(self, from_id: str, to_id: str, relation_type: str) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=digest_source_id("duolingo-progress-csv-edge", from_id, to_id, relation_type),
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.DUOLINGO_PROGRESS_CSV.value,
                "relation_type": relation_type,
            },
        )

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
