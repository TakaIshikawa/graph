"""Adapter for Notion comments CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class NotionCommentsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "notion_comments_csv"

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
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows, start=2):
                unit = self._unit(row, path.name, index)
                if unit is None or (sync_at and unit.updated_at <= sync_at):
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, source_row: int) -> KnowledgeUnit | None:
        comment_id = first(row, "Comment ID", "ID", "comment_id")
        page = first(row, "Page", "Page Title", "Title", "Database")
        text = first(row, "Comment", "Text", "Body", "Content")
        author = first(row, "Author", "Created By", "User")
        created = parse_datetime(first(row, "Created", "Created Time", "Created At", "Date"))
        resolved = first(row, "Resolved", "Is Resolved", "Status")
        url = first(row, "URL", "Page URL", "Link")
        if not any([comment_id, page, text, author, created, url]):
            return None
        now = datetime.now(timezone.utc)
        metadata = clean_metadata({"comment_id": comment_id, "page_title": page, "author": author, "created_at": created.isoformat() if created else first(row, "Created", "Created Time"), "resolved": self._bool(resolved), "url": url, "source_url": url, "external_url": url, "source_file": source_file, "source_row": source_row})
        title = f"Notion comment on {page}" if page else "Notion comment"
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{comment_id}" if comment_id else digest_source_id(self.name, page, text, author, created), source_entity_type="comment", title=title, content="\n".join(part for part in [title, text, f"Author: {author}" if author else "", f"URL: {url}" if url else ""] if part), content_type=ContentType.METADATA, metadata=metadata, tags=list(dict.fromkeys(tag for tag in ["notion", "comment", "resolved" if metadata.get("resolved") else "unresolved" if metadata.get("resolved") is False else ""] if tag)), created_at=created or now, updated_at=created or now)

    def _bool(self, value: Any) -> bool | None:
        text = str(value or "").strip().casefold()
        if text in {"true", "yes", "y", "1", "resolved"}:
            return True
        if text in {"false", "no", "n", "0", "open", "unresolved"}:
            return False
        return None
