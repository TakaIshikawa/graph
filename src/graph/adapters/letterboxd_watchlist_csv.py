"""Adapter for Letterboxd watchlist CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class LetterboxdWatchlistCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "letterboxd_watchlist_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["watchlist_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "watchlist_item" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Name", "name", "Title", "title", "Film", "film")
        year = first(row, "Year", "year", "Release Year", "release_year")
        url = first(row, "Letterboxd URI", "letterboxd_uri", "Letterboxd URL", "URL", "url", "URI", "uri")
        added_text = first(row, "Date", "date", "Date Added", "date_added", "Added", "added_at")
        if not any([title, year, url, added_text]):
            return None

        added_at = parse_datetime(added_text)
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "title": title,
                "year": year,
                "added_date": added_at.isoformat() if added_at else added_text,
                "url": url,
                "letterboxd_uri": url,
                "source_url": url,
                "external_url": url,
                "watchlist": True,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="letterboxd_watchlist_csv",
            source_id=self._source_id(url, title, year, index),
            source_entity_type="watchlist_item",
            title=self._title(title, year),
            content=self._content(title, year, added_at.isoformat() if added_at else added_text, url),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["letterboxd", "watchlist"],
            created_at=added_at or now,
            updated_at=added_at or now,
        )

    def _source_id(self, url: str, title: str, year: str, index: int) -> str:
        return digest_source_id("letterboxd_watchlist_csv", url or title, year, index if not title and not url else "")

    def _title(self, title: str, year: str) -> str:
        if title and year:
            return f"{title} ({year})"
        return title or "Untitled Letterboxd watchlist item"

    def _content(self, title: str, year: str, added_date: str, url: str) -> str:
        parts = []
        if title:
            parts.append(f"Film: {self._title(title, year)}")
        if added_date:
            parts.append(f"Added: {added_date}")
        if url:
            parts.append(f"URL: {url}")
        return "\n".join(parts)
