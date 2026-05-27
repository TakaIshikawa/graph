"""Adapter for Instapaper highlights CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class InstapaperHighlightsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "instapaper_highlights_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["highlight"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "highlight" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit({str(k).strip(): v for k, v in row.items() if k is not None}, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        highlight = first(row, "Highlight", "Text", "Highlighted Text")
        if not highlight:
            return None
        url = first(row, "URL", "Article URL", "Link")
        title = first(row, "Title", "Article Title") or url or "Instapaper highlight"
        note = first(row, "Note", "Notes", "Annotation")
        created_text = first(row, "Created", "Created At", "Date", "Highlighted At")
        created_at = parse_datetime(created_text)
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "url": url,
                "title": title,
                "highlight": highlight,
                "note": note,
                "created_at": created_at.isoformat() if created_at else created_text,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="instapaper_highlights_csv",
            source_id=digest_source_id("instapaper_highlights_csv", url or title or source_file, highlight, index if not url else ""),
            source_entity_type="highlight",
            title=title,
            content="\n".join(part for part in [highlight, f"Note: {note}" if note else "", f"URL: {url}" if url else ""] if part),
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            created_at=created_at or now,
            updated_at=created_at or now,
        )
