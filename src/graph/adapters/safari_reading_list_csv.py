"""Adapter for Safari Reading List CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class SafariReadingListCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "safari_reading_list_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["reading_list_entry"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "reading_list_entry" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        url = first(row, "url", "link", "address")
        title = first(row, "title", "name")
        preview = first(row, "preview text", "preview", "description", "excerpt", "summary")
        if not url and not title and not preview:
            return None
        added_text = first(row, "added date", "date added", "added_at", "created_at", "time_added")
        added_at = parse_datetime(added_text)
        read = _read_status(first(row, "read status", "read", "is_read", "status"))
        folder = first(row, "folder", "source", "collection")
        site_name = first(row, "site name", "site", "domain") or _domain(url)
        metadata = clean_metadata(
            {
                "title": title or url,
                "url": url,
                "source_url": url,
                "preview_text": preview,
                "added_at": added_at.isoformat() if added_at else added_text,
                "read": read,
                "status": "read" if read else "unread",
                "folder": folder,
                "site_name": site_name,
                "source_file": source_file,
            }
        )
        now = datetime.now(timezone.utc)
        display_title = title or site_name or url or "Safari reading list entry"
        return KnowledgeUnit(
            source_project=SourceProject.SAFARI_READING_LIST_CSV,
            source_id=digest_source_id("safari_reading_list_csv", url, added_at.isoformat() if added_at else added_text, title, index),
            source_entity_type="reading_list_entry",
            title=display_title,
            content=_content(display_title, url, preview, metadata),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=[tag for tag in dict.fromkeys(["safari", "reading_list", metadata["status"], folder, site_name]) if tag],
            created_at=added_at or now,
            updated_at=added_at or now,
        )


def _read_status(value: str) -> bool:
    text = value.strip().casefold()
    return text in {"1", "true", "yes", "y", "read", "archived"}


def _domain(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return (parsed.netloc or parsed.path).casefold().removeprefix("www.")


def _content(title: str, url: str, preview: str, metadata: dict[str, Any]) -> str:
    parts = [title, f"URL: {url}" if url else "", preview]
    if metadata.get("folder"):
        parts.append(f"Folder: {metadata['folder']}")
    parts.append(f"Status: {metadata['status']}")
    return "\n".join(part for part in parts if part)
