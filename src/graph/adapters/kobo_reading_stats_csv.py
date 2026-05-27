"""Adapter for Kobo reading statistics CSV exports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, parse_float, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class KoboReadingStatsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "kobo_reading_stats_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["reading_stat"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "reading_stat" not in set(entity_types or self.entity_types):
            return result
        sync_at = since.last_sync_at.astimezone(timezone.utc) if since else None
        for path in iter_paths(self.path, {".csv"}):
            for row in read_csv_rows(path):
                unit = self._unit(row, path)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], path: Path) -> KnowledgeUnit | None:
        title = first(row, "Book Title", "Title")
        author = first(row, "Author", "Authors")
        if not title and not author:
            return None
        last_read = parse_datetime(first(row, "Last Read Date", "Last Read", "Date Last Read")) or datetime.now(timezone.utc)
        percent = parse_float(first(row, "Percent Read", "% Read", "Progress"))
        metadata = clean_metadata(
            {
                "book_title": title,
                "author": author,
                "percent_read": percent,
                "minutes_read": parse_float(first(row, "Minutes Read", "Reading Minutes", "Time Read")),
                "last_read_at": last_read.isoformat(),
                "status": first(row, "Status", "Reading Status"),
                "isbn": first(row, "ISBN", "ISBN13"),
                "shelves": split_values(first(row, "Shelves", "Shelf", "Collections")),
                "source_file": str(path),
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project=self.name,
            source_id=digest_source_id(self.name, first(row, "ISBN", "ISBN13") or title.casefold(), author.casefold()),
            source_entity_type="reading_stat",
            title=f"Kobo reading: {title}" if title else f"Kobo reading: {author}",
            content="\n".join(part for part in (f"Book: {title}" if title else "", f"Author: {author}" if author else "", f"Percent read: {percent}" if percent is not None else "", f"Last read: {last_read.isoformat()}") if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["kobo", "reading_stat"],
            created_at=last_read,
            updated_at=last_read,
        )
