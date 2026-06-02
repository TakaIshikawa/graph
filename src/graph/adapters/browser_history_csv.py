"""Adapter for browser history CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class BrowserHistoryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "browser_history_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["web_history"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "web_history" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        url = first(row, "url", "uri", "link")
        if not url:
            return None
        visited_text = first(row, "visited_at", "last_visit_time", "last visited", "visit_time", "timestamp")
        visited_at = parse_datetime(visited_text) or datetime.now(timezone.utc)
        title = first(row, "title", "page_title", "name") or _title_from_url(url)
        visit_count = first(row, "visit_count", "visits")
        metadata = clean_metadata(
            {
                "url": url,
                "visit_count": visit_count,
                "visited_at": visited_text,
                "last_visit_time": first(row, "last_visit_time", "last visited"),
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="browser_history_csv",
            source_id=digest_source_id("browser_history_csv", url, visited_text or index),
            source_entity_type="web_history",
            title=title,
            content=_content(title, url),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            created_at=visited_at,
            updated_at=visited_at,
        )


def _title_from_url(url: str) -> str:
    parsed = urlsplit(url if "://" in url else f"https://{url}")
    host = parsed.netloc or parsed.path.split("/", 1)[0]
    path = parsed.path if parsed.netloc else "/" + parsed.path.split("/", 1)[1] if "/" in parsed.path else ""
    fallback = f"{host}{path}".rstrip("/")
    return fallback or url


def _content(title: str, url: str) -> str:
    return "\n".join(part for part in (title, f"URL: {url}") if part)
