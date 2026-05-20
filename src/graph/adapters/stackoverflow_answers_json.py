"""Adapter for Stack Overflow answer JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class StackOverflowAnswersJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "stackoverflow_answers_json"

    @property
    def entity_types(self) -> list[str]:
        return ["answer"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else set(self.entity_types)
        if "answer" not in allowed:
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        units: dict[str, KnowledgeUnit] = {}
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
                units[unit.source_id] = unit

        result.units.extend(sorted(units.values(), key=lambda unit: (unit.updated_at, unit.source_id)))
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._records(parsed)

    def _records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []
        for key in ("answers", "items", "data", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                records = self._records(nested)
                if records:
                    return records
        return [value] if self._looks_like_record(value) else []

    def _looks_like_record(self, record: dict[str, Any]) -> bool:
        return bool(first(record, "answer_id", "answerId", "id") or first(record, "body", "answer_body", "body_markdown", "link"))

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        answer_id = first(record, "answer_id", "answerId", "id")
        question_id = first(record, "question_id", "questionId")
        question_title = first(record, "question_title", "questionTitle", "title")
        body = first(record, "body", "answer_body", "body_markdown", "bodyMarkdown")
        url = first(record, "url", "link", "answer_url", "answerUrl")
        if not any([answer_id, question_id, question_title, body, url]):
            return None

        created = self._record_datetime(record, "creation_date", "creationDate", "created_at", "createdAt")
        updated = self._record_datetime(record, "last_edit_date", "lastEditDate", "updated_at", "updatedAt", "last_activity_date", "lastActivityDate") or created
        tags = split_values(record.get("tags") or record.get("tag_names") or record.get("tagNames"))
        owner = self._owner(record.get("owner") or record.get("user"))
        event_at = updated or created or datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "answer_id": parse_int(answer_id) if answer_id else None,
                "question_id": parse_int(question_id) if question_id else None,
                "question_title": question_title,
                "score": parse_int(record.get("score")),
                "is_accepted": self._parse_bool(record.get("is_accepted") if "is_accepted" in record else record.get("accepted")),
                "url": url,
                "source_url": url,
                "tags": tags,
                "owner": owner,
                "created_at": created.isoformat() if created else None,
                "updated_at": updated.isoformat() if updated else None,
                "license": first(record, "license", "content_license", "contentLicense"),
                "source_file": source_file,
                "raw": dict(record),
            }
        )
        title = question_title or f"Stack Overflow answer {answer_id or question_id or url}"
        return KnowledgeUnit(
            source_project="stackoverflow_answers_json",
            source_id=self._source_id(answer_id, question_id, url, body),
            source_entity_type="answer",
            title=title,
            content=self._content(title, body, metadata),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["stackoverflow", "answer", *tags],
            created_at=created or event_at,
            updated_at=event_at,
        )

    def _source_id(self, answer_id: str, question_id: str, url: str, body: str) -> str:
        if answer_id:
            return f"stackoverflow_answers_json:answer:{answer_id}"
        return digest_source_id("stackoverflow_answers_json", question_id, url, body)

    def _owner(self, value: Any) -> str | dict[str, Any]:
        if isinstance(value, dict):
            owner = clean_metadata(
                {
                    "display_name": first(value, "display_name", "displayName", "name"),
                    "user_id": parse_int(first(value, "user_id", "userId", "account_id", "accountId")),
                    "link": first(value, "link", "url"),
                }
            )
            return owner or ""
        return "" if value is None else str(value).strip()

    def _record_datetime(self, record: dict[str, Any], *keys: str) -> datetime | None:
        for key in keys:
            parsed = self._parse_datetime_value(record.get(key))
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
        if text in {"false", "no", "n", "0"}:
            return False
        return None

    def _content(self, title: str, body: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        if body:
            parts.append(body)
        for label, key in (
            ("Score", "score"),
            ("Accepted", "is_accepted"),
            ("URL", "url"),
            ("License", "license"),
        ):
            if key in metadata:
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
