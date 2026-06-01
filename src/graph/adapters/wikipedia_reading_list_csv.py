"""Adapter for Wikipedia app reading-list CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class WikipediaReadingListCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "wikipedia_reading_list_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["article"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "article" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row_number, row in enumerate(rows, start=1):
                unit = self._unit(row, path.name, row_number)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _read_rows(self, path: Any) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            return [{str(key).strip(): "" if value is None else str(value).strip() for key, value in row.items() if key is not None} for row in reader]

    def _unit(self, row: dict[str, Any], source_file: str, row_number: int) -> KnowledgeUnit | None:
        title = first(row, "Title", "Page title", "Article", "Name")
        url = first(row, "URL", "Url", "Page URL", "Article URL")
        if not title and not url:
            return None
        wiki = first(row, "Wiki", "Project", "Site")
        language = first(row, "Language", "Lang")
        list_name = first(row, "List", "Reading List", "Folder")
        description = first(row, "Description", "Extract", "Summary")
        saved_text = first(row, "Saved", "Saved at", "Saved timestamp", "Created", "Date")
        saved_at = parse_datetime(saved_text)
        archived = self._bool(first(row, "Archived", "Archive"))
        read = self._bool(first(row, "Read", "Read status", "Completed"))
        metadata = clean_metadata(
            {
                "title": title,
                "url": url,
                "wiki": wiki,
                "language": language,
                "list": list_name,
                "description": description,
                "saved_at": saved_at.isoformat() if saved_at else saved_text,
                "archived": archived,
                "read": read,
                "source_file": source_file,
                "row_number": row_number,
            }
        )
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project="wikipedia_reading_list_csv",
            source_id=f"wikipedia_reading_list_csv:{url}" if url else digest_source_id("wikipedia_reading_list_csv", title, wiki, language, list_name),
            source_entity_type="article",
            title=title or url,
            content=self._content(title, url, description, language, list_name),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(["wikipedia", "article", language, list_name] if list_name else ["wikipedia", "article", language])),
            created_at=saved_at or now,
            updated_at=saved_at or now,
        )

    def _content(self, title: str, url: str, description: str, language: str, list_name: str) -> str:
        parts = [title, f"URL: {url}" if url else "", description, f"Language: {language}" if language else "", f"List: {list_name}" if list_name else ""]
        return "\n".join(part for part in parts if part)

    def _bool(self, value: str) -> bool | None:
        text = value.strip().casefold()
        if text in {"1", "true", "yes", "y", "read", "archived"}:
            return True
        if text in {"0", "false", "no", "n", "unread", "active"}:
            return False
        return None
