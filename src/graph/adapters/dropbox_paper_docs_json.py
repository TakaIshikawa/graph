"""Adapter for Dropbox Paper document JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class DropboxPaperDocsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "dropbox_paper_docs_json"

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
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit(record, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("documents", "docs", "paper_docs", "data", "results"):
                if isinstance(parsed.get(key), list):
                    return [item for item in parsed[key] if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        doc_id = self._text(record.get("id") or record.get("doc_id") or record.get("paper_doc_id"))
        title = self._text(record.get("title") or record.get("name"))
        body = self._text(record.get("body") or record.get("markdown") or record.get("text") or record.get("content"))
        url = self._text(record.get("sharing_url") or record.get("url") or record.get("doc_url"))
        if not any([doc_id, title, body]):
            return None
        created = parse_datetime(record.get("created") or record.get("created_at") or record.get("created_time"))
        updated = parse_datetime(record.get("updated") or record.get("updated_at") or record.get("modified_time")) or created
        owner = self._person(record.get("owner") or record.get("created_by"))
        folder = self._text(record.get("folder") or record.get("path") or record.get("folder_path"))
        tags = split_values(record.get("tags"))
        metadata = clean_metadata(
            {
                "document_id": doc_id,
                "title": title,
                "body": body,
                "owner": owner,
                "created_at": created.isoformat() if created else self._text(record.get("created_at")),
                "updated_at": updated.isoformat() if updated else self._text(record.get("updated_at")),
                "sharing_url": url,
                "source_url": url,
                "folder": folder,
                "tags": tags,
                "source_file": source_file,
            }
        )
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.DROPBOX_PAPER_DOCS_JSON,
            source_id=f"dropbox_paper_docs_json:{doc_id}" if doc_id else digest_source_id("dropbox_paper_docs_json", title, body, url, index),
            source_entity_type="document",
            title=title or doc_id or "Dropbox Paper document",
            content="\n".join(part for part in [title, body, f"URL: {url}" if url else "", f"Folder: {folder}" if folder else ""] if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(["dropbox", "paper", "document", *tags])),
            created_at=created or updated or now,
            updated_at=updated or created or now,
        )

    def _person(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("email") or value.get("display_name") or value.get("name") or value.get("account_id"))
        return self._text(value)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
