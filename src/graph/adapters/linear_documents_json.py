"""Adapter for Linear document JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class LinearDocumentsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "linear_documents_json"

    @property
    def entity_types(self) -> list[str]:
        return ["document"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "document" not in entity_types:
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

    def _records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("documents", "items", "nodes"):
                if isinstance(value.get(key), (dict, list)):
                    records = self._records(value[key])
                    if records:
                        return records
            data = value.get("data")
            if isinstance(data, dict):
                return self._records(data)
            return [value] if first(value, "id", "title", "content", "body") else []
        return []

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(record, "title", "name")
        content = first(record, "content", "body", "markdown", "text")
        doc_id = first(record, "id", "documentId")
        if not any((title, content, doc_id)):
            return None
        creator = self._name(record.get("creator")) or first(record, "creator", "createdBy")
        project = self._name(record.get("project")) or first(record, "project", "projectName", "projectId")
        team = self._name(record.get("team")) or first(record, "team", "teamName", "teamId")
        url = first(record, "url", "appUrl")
        created_at = parse_datetime(first(record, "createdAt", "created_at", "created"))
        updated_at = parse_datetime(first(record, "updatedAt", "updated_at", "modified")) or created_at or datetime.now(timezone.utc)
        archived = self._bool(record.get("archived"))
        metadata = clean_metadata({"document_id": doc_id, "title": title, "creator": creator, "project": project, "team": team, "url": url, "archived": archived, "created_at": created_at.isoformat() if created_at else None, "updated_at": updated_at.isoformat(), "source_file": source_file})
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{doc_id}" if doc_id else digest_source_id(self.name, title, content[:80], index), source_entity_type="document", title=title or doc_id or "Linear document", content=self._content(title, content, project, team, url), content_type=ContentType.ARTIFACT, metadata=metadata, tags=["linear", "document"], created_at=created_at or updated_at, updated_at=updated_at)

    def _name(self, value: Any) -> str:
        return first(value, "name", "title", "id", "key") if isinstance(value, dict) else ""

    def _bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().casefold()
        if text in {"true", "1", "yes", "archived"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        return None

    def _content(self, title: str, body: str, project: str, team: str, url: str) -> str:
        return "\n".join(part for part in (title, body, f"Project: {project}" if project else "", f"Team: {team}" if team else "", f"URL: {url}" if url else "") if part)
