"""Adapter for Google Drive comments JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GoogleDriveCommentsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_drive_comments_json"

    @property
    def entity_types(self) -> list[str]:
        return ["comment"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "comment" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._records(json.loads(path.read_text(encoding="utf-8-sig")))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit(record, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _records(self, value: Any, file_meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [record for item in value for record in self._records(item, file_meta)]
        if not isinstance(value, dict):
            return []
        current_file = value.get("file") if isinstance(value.get("file"), dict) else file_meta
        if any(key in value for key in ("comment", "content", "text", "htmlContent", "replies")) and not any(key in value for key in ("comments", "items")):
            record = dict(value)
            if current_file:
                record.setdefault("file", current_file)
            return [record]
        records = []
        for key in ("comments", "items", "results", "data"):
            if isinstance(value.get(key), (dict, list)):
                records.extend(self._records(value[key], current_file or value))
        return records

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        body = first(record, "comment", "content", "text", "htmlContent")
        replies = self._replies(record.get("replies"))
        if not body and not replies:
            return None
        file_meta = record.get("file") if isinstance(record.get("file"), dict) else {}
        file_id = first(record, "fileId", "file_id") or first(file_meta, "id", "fileId")
        file_title = first(record, "fileTitle", "file_title", "title") or first(file_meta, "title", "name")
        file_url = first(record, "fileUrl", "file_url", "url", "webUrl") or first(file_meta, "url", "webUrl", "alternateLink")
        author = self._author(record.get("author")) or first(record, "author", "authorName")
        quoted = first(record, "quotedText", "quoted_text", "context")
        created_at = parse_datetime(first(record, "createdTime", "created_at", "created"))
        updated_at = parse_datetime(first(record, "modifiedTime", "updated_at", "modified")) or created_at or datetime.now(timezone.utc)
        resolved = self._bool(record.get("resolved"))
        metadata = clean_metadata({"comment_id": first(record, "id", "commentId"), "file_id": file_id, "file_title": file_title, "file_url": file_url, "author": author, "quoted_text": quoted, "replies": replies, "resolved": resolved, "created_at": created_at.isoformat() if created_at else None, "updated_at": updated_at.isoformat(), "source_file": source_file})
        return KnowledgeUnit(source_project=self.name, source_id=digest_source_id(self.name, first(record, "id", "commentId") or file_id or file_url, body, index), source_entity_type="comment", title=file_title or body[:80] or "Google Drive comment", content=self._content(body, quoted, author, file_title, file_url, replies, resolved), content_type=ContentType.ARTIFACT, metadata=metadata, tags=["google_drive", "comment"], created_at=created_at or updated_at, updated_at=updated_at)

    def _author(self, value: Any) -> str:
        return first(value, "displayName", "name", "emailAddress", "email") if isinstance(value, dict) else ""

    def _replies(self, value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        replies = []
        for reply in value:
            if isinstance(reply, dict):
                text = first(reply, "content", "text", "htmlContent")
                if text:
                    replies.append(clean_metadata({"text": text, "author": self._author(reply.get("author")) or first(reply, "author"), "created_at": first(reply, "createdTime", "created_at", "created")}))
        return replies

    def _bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().casefold()
        if text in {"true", "1", "yes", "resolved"}:
            return True
        if text in {"false", "0", "no", "open"}:
            return False
        return None

    def _content(self, body: str, quoted: str, author: str, file_title: str, file_url: str, replies: list[dict[str, str]], resolved: bool | None) -> str:
        parts = [body, f"Quoted: {quoted}" if quoted else "", f"Author: {author}" if author else "", f"File: {file_title}" if file_title else "", f"URL: {file_url}" if file_url else ""]
        if resolved is not None:
            parts.append(f"Resolved: {resolved}")
        parts.extend(f"Reply: {reply['text']}" for reply in replies)
        return "\n".join(part for part in parts if part)
