"""Adapter for Pinterest saved pins CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class PinterestPinsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pinterest_pins_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["pin"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "pin" not in entity_types:
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
        title = first(row, "title", "Title", "name", "Name", "pin_title", "Pin Title")
        description = first(row, "description", "Description", "desc", "summary", "note", "notes")
        board = first(row, "board", "Board", "board_name", "Board Name", "collection", "Collection")
        saved_text = first(row, "saved_at", "Saved At", "saved_date", "Saved Date", "created_at", "Created At", "created", "Created", "date", "Date")
        link = first(row, "link", "Link", "url", "URL", "source_url", "Source URL", "domain_url", "Domain URL")
        image_url = first(row, "image_url", "Image URL", "image", "Image", "media_url", "Media URL")
        pin_url = first(row, "pin_url", "Pin URL", "pinterest_url", "Pinterest URL", "pin_link", "Pin Link")
        if not any([title, description, board, saved_text, link, image_url, pin_url]):
            return None

        saved_at = parse_datetime(saved_text)
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "title": title or pin_url or link,
                "description": description,
                "board": board,
                "saved_at": saved_at.isoformat() if saved_at else saved_text,
                "link": link,
                "url": link or pin_url,
                "source_url": link or pin_url,
                "external_url": link or pin_url,
                "image_url": image_url,
                "pin_url": pin_url,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="pinterest_pins_csv",
            source_id=self._source_id(pin_url, link, title, board, index),
            source_entity_type="pin",
            title=title or pin_url or link or "Untitled Pinterest pin",
            content=self._content(title, description, board, saved_at.isoformat() if saved_at else saved_text, link, image_url, pin_url),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=self._tags(board),
            created_at=saved_at or now,
            updated_at=saved_at or now,
        )

    def _source_id(self, pin_url: str, link: str, title: str, board: str, index: int) -> str:
        return digest_source_id("pinterest_pins_csv", pin_url or link or title, board, index if not any([pin_url, link, title]) else "")

    def _tags(self, board: str) -> list[str]:
        tags = ["pinterest", "pin", "bookmark"]
        normalized_board = " ".join(board.casefold().split())
        if normalized_board:
            tags.append(normalized_board)
        return tags

    def _content(self, title: str, description: str, board: str, saved_at: str, link: str, image_url: str, pin_url: str) -> str:
        parts = [title] if title else []
        if description:
            parts.append(f"Description: {description}")
        if board:
            parts.append(f"Board: {board}")
        if saved_at:
            parts.append(f"Saved: {saved_at}")
        if link:
            parts.append(f"URL: {link}")
        if image_url:
            parts.append(f"Image: {image_url}")
        if pin_url:
            parts.append(f"Pin: {pin_url}")
        return "\n".join(parts)
