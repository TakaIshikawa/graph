"""Adapter for Coursera progress CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class CourseraProgressCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "coursera_progress_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["course_progress"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "course_progress" not in set(entity_types or self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.csv") if child.is_file())

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        course = self._first(row, "course", "course_title", "Course", "Course Title")
        specialization = self._first(row, "specialization", "Specialization")
        item = self._first(row, "item", "lesson", "module", "Item", "Lesson", "Module")
        lesson = self._first(row, "lesson", "Lesson")
        module = self._first(row, "module", "Module")
        progress = self._parse_number(self._first(row, "progress", "Progress", "Percent Complete"))
        grade = self._parse_number(self._first(row, "grade", "Grade", "Score"))
        status = self._first(row, "status", "Status")
        started_at = self._parse_datetime(self._first(row, "started_at", "Started At", "Start Date"))
        completed_at = self._parse_datetime(self._first(row, "completed_at", "Completed At", "Completion Date"))
        updated_at = self._parse_datetime(self._first(row, "updated_at", "Updated At", "Last Updated")) or completed_at or started_at
        certificate_url = self._first(row, "certificate_url", "Certificate URL", "Certificate")
        course_url = self._first(row, "course_url", "Course URL", "URL")
        if not course and not item and not course_url and not certificate_url:
            return None
        metadata = {
            "course": course,
            "specialization": specialization,
            "item": item,
            "lesson": lesson,
            "module": module,
            "progress": progress,
            "grade": grade,
            "status": status,
            "started_at": started_at.isoformat() if started_at else self._first(row, "started_at", "Started At", "Start Date"),
            "completed_at": completed_at.isoformat() if completed_at else self._first(row, "completed_at", "Completed At", "Completion Date"),
            "updated_at": updated_at.isoformat() if updated_at else self._first(row, "updated_at", "Updated At", "Last Updated"),
            "certificate_url": certificate_url,
            "course_url": course_url,
            "source_file": source_file,
            "row": dict(row),
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.COURSERA_PROGRESS_CSV,
            source_id=self._source_id(course, item, course_url, certificate_url),
            source_entity_type="course_progress",
            title=item or course or course_url or certificate_url,
            content=self._content(course, specialization, item, status, course_url),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(item for item in ["coursera", "course_progress", course, specialization] if item)),
            created_at=started_at or completed_at or updated_at or now,
            updated_at=updated_at or now,
        )

    def _content(self, course: str, specialization: str, item: str, status: str, url: str) -> str:
        parts = [item or course, f"Course: {course}" if course and item else "", f"Specialization: {specialization}" if specialization else "", f"Status: {status}" if status else "", f"URL: {url}" if url else ""]
        return "\n".join(part for part in parts if part)

    def _source_id(self, course: str, item: str, course_url: str, certificate_url: str) -> str:
        digest = hashlib.sha256((certificate_url or course_url or f"{course}|{item}").encode("utf-8")).hexdigest()[:24]
        return f"coursera_progress_csv:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_number(self, value: str) -> float | int | None:
        text = value.strip().rstrip("%")
        if not text:
            return None
        try:
            number = float(text)
            return int(number) if number.is_integer() else number
        except ValueError:
            return None

    def _parse_datetime(self, value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
