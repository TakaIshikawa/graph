"""Adapter for Libby holds CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class LibbyHoldsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "libby_holds_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["libby_hold"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types and "libby_hold" not in entity_types:
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit(row, path.name)
                if unit is None:
                    continue
                placed_at = self._placed_at(unit)
                if sync_at and placed_at and placed_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = first(row, "Title", "Name")
        authors = split_values(first(row, "Author", "Authors", "Creator"))
        hold_id = first(row, "Hold ID", "Libby Hold ID", "ID")
        placed_raw = first(row, "Placed Date", "Placed", "Placed At", "Hold Placed Date")
        suspended_raw = first(row, "Suspended Until", "Suspend Until", "Suspension End Date")
        placed_at = parse_datetime(placed_raw)
        suspended_until = parse_datetime(suspended_raw)
        source_url = first(row, "Source URL", "URL", "Link", "Title URL")
        if not any([title, authors, hold_id, source_url]):
            return None

        metadata = {
            "title": title,
            "authors": authors,
            "format": first(row, "Format", "Media Format", "Type"),
            "library": first(row, "Library", "Library Name"),
            "estimated_wait": first(row, "Estimated Wait", "Wait Time", "Estimated Wait Time"),
            "estimated_wait_days": self._wait_days(first(row, "Estimated Wait", "Wait Time", "Estimated Wait Time")),
            "queue_position": parse_int(first(row, "Queue Position", "Position", "Place in Line", "You Are #")),
            "placed_at": placed_at.isoformat() if placed_at else placed_raw,
            "suspended_until": suspended_until.isoformat() if suspended_until else suspended_raw,
            "source_url": source_url,
            "hold_id": hold_id,
            "source_file": source_file,
        }
        now = datetime.now(timezone.utc)
        name = title or "Libby hold"
        return KnowledgeUnit(
            source_project="libby_holds_csv",
            source_id=self._source_id(title, authors, placed_at, hold_id),
            source_entity_type="libby_hold",
            title=name,
            content=self._content(name, metadata),
            content_type=ContentType.METADATA,
            metadata=clean_metadata(metadata),
            tags=list(dict.fromkeys(tag for tag in ["libby", "hold", metadata["format"], metadata["library"], *authors] if tag)),
            created_at=placed_at or now,
            updated_at=placed_at or now,
        )

    def _source_id(self, title: str, authors: list[str], placed_at: datetime | None, hold_id: str) -> str:
        if hold_id:
            return digest_source_id("libby_holds_csv", hold_id)
        return digest_source_id("libby_holds_csv", title, ";".join(authors), placed_at.isoformat() if placed_at else "")

    def _content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        if metadata.get("authors"):
            parts.append(f"Authors: {', '.join(metadata['authors'])}")
        for key, label in (
            ("format", "Format"),
            ("library", "Library"),
            ("estimated_wait", "Estimated wait"),
            ("queue_position", "Queue position"),
            ("placed_at", "Placed"),
            ("suspended_until", "Suspended until"),
            ("source_url", "URL"),
        ):
            if metadata.get(key) not in ("", None, []):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)

    def _wait_days(self, value: str) -> int | None:
        text = value.strip().lower()
        if not text:
            return None
        number = parse_int(text)
        if number is None:
            return None
        if "month" in text:
            return number * 30
        if "week" in text or re.search(r"\bwks?\b", text):
            return number * 7
        if "hour" in text:
            return 1 if number > 0 else 0
        return number

    def _placed_at(self, unit: KnowledgeUnit) -> datetime | None:
        value = unit.metadata.get("placed_at")
        return parse_datetime(value)
