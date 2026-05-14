"""Adapter for Stack Overflow bookmarked question JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import (
    clean_metadata,
    digest_source_id,
    ensure_utc,
    first,
    iter_paths,
    parse_datetime,
    parse_int,
    split_values,
)
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class StackOverflowBookmarksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "stackoverflow_bookmarks_json"

    @property
    def entity_types(self) -> list[str]:
        return ["question_bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else set(self.entity_types)
        if "question_bookmark" not in allowed:
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for record in records:
                unit = self._unit_from_record(record, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.updated_at, unit.source_id)))
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._records(parsed)

    def _records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []
        for key in ("items", "bookmarks", "saved", "questions", "data", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                records = self._records(nested)
                if records:
                    return records
        return [value] if self._looks_like_record(value) else []

    def _looks_like_record(self, value: dict[str, Any]) -> bool:
        return bool(
            first(value, "title", "question_title", "link")
            or first(value, "question_id", "questionId", "id")
            or first(value, "url", "link", "question_url")
        )

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = first(record, "title", "question_title", "questionTitle")
        url = first(record, "url", "link", "question_url", "questionUrl")
        question_id = first(record, "question_id", "questionId", "id")
        if not title and not url:
            return None

        created_at = self._record_datetime(
            record,
            "creation_date",
            "creationDate",
            "created_at",
            "createdAt",
        )
        bookmarked_at = self._record_datetime(
            record,
            "bookmark_date",
            "bookmarked_at",
            "bookmarkedAt",
            "saved_at",
            "savedAt",
            "favorited_at",
            "favoritedAt",
            "updated_at",
            "updatedAt",
            "last_activity_date",
            "lastActivityDate",
        )
        updated_at = bookmarked_at or self._record_datetime(
            record,
            "updated_at",
            "updatedAt",
            "last_activity_date",
            "lastActivityDate",
        ) or created_at
        now = datetime.now(timezone.utc)
        event_at = updated_at or created_at or now
        tags = split_values(record.get("tags") or record.get("tag_names") or record.get("tagNames"))
        metadata = clean_metadata(
            {
                "question_id": parse_int(question_id) if question_id else None,
                "title": title,
                "url": url,
                "tags": tags,
                "score": parse_int(record.get("score")),
                "answer_count": parse_int(record.get("answer_count") or record.get("answerCount")),
                "accepted_answer": self._parse_bool(
                    record.get("accepted_answer")
                    if "accepted_answer" in record
                    else record.get("acceptedAnswer", record.get("is_answered"))
                ),
                "creation_date": created_at.isoformat() if created_at else None,
                "bookmark_date": bookmarked_at.isoformat() if bookmarked_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
                "source_file": source_file,
                "record": dict(record),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.STACKOVERFLOW_BOOKMARKS_JSON,
            source_id=self._source_id(question_id, url, title),
            source_entity_type="question_bookmark",
            title=title or url,
            content=self._content(title, url, tags, metadata),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["stackoverflow", "question_bookmark", *tags],
            created_at=created_at or event_at,
            updated_at=event_at,
        )

    def _source_id(self, question_id: str, url: str, title: str) -> str:
        if question_id:
            return f"stackoverflow_bookmarks_json:question:{question_id}"
        return digest_source_id("stackoverflow_bookmarks_json", url or title)

    def _content(self, title: str, url: str, tags: list[str], metadata: dict[str, Any]) -> str:
        parts = [title] if title else []
        if url:
            parts.append(f"URL: {url}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        for label, key in (
            ("Score", "score"),
            ("Answers", "answer_count"),
            ("Accepted answer", "accepted_answer"),
        ):
            if key in metadata:
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)

    def _record_datetime(self, record: dict[str, Any], *keys: str) -> datetime | None:
        for key in keys:
            value = record.get(key)
            parsed = self._parse_datetime_value(value)
            if parsed is not None:
                return parsed
        return None

    def _parse_datetime_value(self, value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        text = str(value).strip()
        if text.isdigit():
            try:
                return datetime.fromtimestamp(float(text), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        return parse_datetime(text)

    def _parse_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if value in (None, ""):
            return None
        text = str(value).strip().casefold()
        if text in {"true", "yes", "y", "1", "accepted"}:
            return True
        if text in {"false", "no", "n", "0", "none"}:
            return False
        return None
