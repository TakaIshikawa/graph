"""Adapter for Coursera course CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class CourseraCoursesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "coursera_courses_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["course"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "course" not in entity_types:
            return result
        sync_at = since.last_sync_at.astimezone(timezone.utc) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Course Title", "Title", "Course")
        provider = first(row, "Provider", "University", "Institution", "Partner")
        instructor = first(row, "Instructor", "Instructors")
        status = first(row, "Status", "Enrollment Status", "Completion Status")
        progress = parse_float(first(row, "Progress", "Progress %", "Percent Complete"))
        certificate_url = first(row, "Certificate URL", "Certificate", "Certificate Link")
        completed_at = parse_datetime(first(row, "Completed Date", "Completed At", "Completion Date"))
        enrolled_at = parse_datetime(first(row, "Enrollment Date", "Enrolled At", "Started At"))
        course_url = first(row, "Course URL", "URL", "Url", "Link")
        if not any([title, provider, course_url]):
            return None
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "course_title": title,
                "provider": provider,
                "instructor": instructor,
                "status": status,
                "progress": progress,
                "certificate_url": certificate_url,
                "completed_at": completed_at.isoformat() if completed_at else "",
                "enrolled_at": enrolled_at.isoformat() if enrolled_at else "",
                "course_url": course_url,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="coursera_courses_csv",
            source_id=digest_source_id("coursera_courses_csv", course_url or title, provider or index),
            source_entity_type="course",
            title=title or course_url or "Coursera course",
            content=self._content(title, provider, instructor, status, progress, course_url, certificate_url),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["coursera", "course", provider, status] if tag)),
            created_at=enrolled_at or completed_at or now,
            updated_at=completed_at or enrolled_at or now,
        )

    def _content(self, title: str, provider: str, instructor: str, status: str, progress: float | None, course_url: str, certificate_url: str) -> str:
        parts = [title]
        for label, value in (("Provider", provider), ("Instructor", instructor), ("Status", status), ("Progress", progress), ("Course URL", course_url), ("Certificate URL", certificate_url)):
            if value not in ("", None):
                parts.append(f"{label}: {value}")
        return "\n".join(part for part in parts if part)
