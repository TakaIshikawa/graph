"""Adapter for Kindle My Clippings.txt exports."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState

_DELIMITER = "=========="


class KindleClippingsTxtAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "kindle_clippings_txt"

    @property
    def entity_types(self) -> list[str]:
        return ["clipping"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "clipping" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        if sync_at and sync_at.tzinfo is None:
            sync_at = sync_at.replace(tzinfo=timezone.utc)

        for path in self._iter_paths():
            try:
                text = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError):
                continue
            for index, block in enumerate(text.split(_DELIMITER)):
                unit = self._unit_from_block(block, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".txt":
            return [root]
        if root.is_dir():
            return sorted(path for path in root.rglob("*.txt") if path.is_file())
        return []

    def _unit_from_block(self, block: str, source_file: str, index: int) -> KnowledgeUnit | None:
        lines = [line.strip() for line in block.strip().splitlines()]
        lines = [line for line in lines if line]
        if len(lines) < 3:
            return None
        title, author = self._title_author(lines[0])
        details = lines[1].lstrip("- ").strip()
        body = "\n".join(lines[2:]).strip()
        if not body:
            return None

        clipping_type = self._clipping_type(details)
        location = self._match(details, r"Location\s+([^|]+)")
        page = self._match(details, r"page\s+([^|]+)")
        added_text = self._match(details, r"Added on\s+(.+)$")
        added_at = parse_datetime(added_text)
        now = datetime.now(timezone.utc)

        metadata = clean_metadata(
            {
                "book_title": title,
                "author": author,
                "clipping_type": clipping_type,
                "location": location,
                "page": page,
                "added": added_text,
                "added_at": added_at.isoformat() if added_at else None,
                "body": body,
                "details": details,
                "source_file": source_file,
                "record_index": index,
            }
        )
        return KnowledgeUnit(
            source_project="kindle_clippings_txt",
            source_id=self._source_id(title, author, clipping_type, location, page, added_text, body, index),
            source_entity_type=clipping_type or "clipping",
            title=title or "Kindle clipping",
            content=self._content(body, title, author, clipping_type, location, page, added_text),
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=["kindle", clipping_type] if clipping_type else ["kindle"],
            created_at=added_at or now,
            updated_at=added_at or now,
        )

    def _title_author(self, value: str) -> tuple[str, str]:
        match = re.match(r"^(?P<title>.+?)\s+\((?P<author>[^()]+)\)$", value)
        if match:
            return match.group("title").strip(), match.group("author").strip()
        return value.strip(), ""

    def _clipping_type(self, details: str) -> str:
        lowered = details.casefold()
        if "note" in lowered:
            return "note"
        if "bookmark" in lowered:
            return "bookmark"
        if "highlight" in lowered:
            return "highlight"
        return "clipping"

    def _match(self, text: str, pattern: str) -> str:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _source_id(self, *parts: Any) -> str:
        digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:24]
        return f"kindle_clippings_txt:{digest}"

    def _content(self, body: str, title: str, author: str, clipping_type: str, location: str, page: str, added: str) -> str:
        parts = [body]
        if title:
            parts.append(f"Title: {title}")
        if author:
            parts.append(f"Author: {author}")
        if clipping_type:
            parts.append(f"Type: {clipping_type}")
        if location:
            parts.append(f"Location: {location}")
        if page:
            parts.append(f"Page: {page}")
        if added:
            parts.append(f"Added: {added}")
        return "\n".join(parts)
