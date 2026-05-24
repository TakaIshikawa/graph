"""Adapter for Audible listening history CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, parse_duration_seconds, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class AudibleListeningHistoryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "audible_listening_history_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["listening_event"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "listening_event" not in entity_types:
            return result
        sync_at = since.last_sync_at.astimezone(timezone.utc) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Title", "Book Title", "Product Name")
        asin = first(row, "ASIN", "Asin")
        started = parse_datetime(first(row, "Started At", "Start Time", "Started", "Date Started"))
        finished = parse_datetime(first(row, "Finished At", "End Time", "Finished", "Date Finished", "Last Listened"))
        duration_text = first(row, "Duration Listened", "Listening Time", "Time Listened", "Duration")
        duration_seconds = self._duration_seconds(duration_text)
        if not any((title, asin, started, finished, duration_text)):
            return None
        authors = split_values(first(row, "Author", "Authors"))
        narrators = split_values(first(row, "Narrator", "Narrators"))
        marketplace = first(row, "Marketplace", "Region")
        device = first(row, "Device", "Device Type", "Platform")
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "title": title,
                "authors": authors,
                "narrators": narrators,
                "asin": asin,
                "started_at": started.isoformat() if started else "",
                "finished_at": finished.isoformat() if finished else "",
                "duration_listened": duration_text,
                "duration_listened_seconds": duration_seconds,
                "marketplace": marketplace,
                "device": device,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="audible_listening_history_csv",
            source_id=digest_source_id("audible_listening_history_csv", asin or title, started.isoformat() if started else "", finished.isoformat() if finished else "", duration_text, index),
            source_entity_type="listening_event",
            title=f"Listened to {title}" if title else "Audible listening event",
            content=self._content(title, authors, narrators, started, finished, duration_text, marketplace, device),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["audible", "listening-event"],
            created_at=started or finished or now,
            updated_at=finished or started or now,
        )

    def _duration_seconds(self, value: str) -> int | None:
        text = value.casefold().strip()
        if not text:
            return None
        hours = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", text)
        minutes = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b", text)
        seconds = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b", text)
        if hours or minutes or seconds:
            total = 0.0
            if hours:
                total += float(hours.group(1)) * 3600
            if minutes:
                total += float(minutes.group(1)) * 60
            if seconds:
                total += float(seconds.group(1))
            return int(round(total))
        return parse_duration_seconds(value)

    def _content(self, title: str, authors: list[str], narrators: list[str], started: datetime | None, finished: datetime | None, duration: str, marketplace: str, device: str) -> str:
        parts = [title] if title else []
        if authors:
            parts.append(f"Author: {', '.join(authors)}")
        if narrators:
            parts.append(f"Narrator: {', '.join(narrators)}")
        if started:
            parts.append(f"Started: {started.isoformat()}")
        if finished:
            parts.append(f"Finished: {finished.isoformat()}")
        if duration:
            parts.append(f"Duration listened: {duration}")
        if marketplace:
            parts.append(f"Marketplace: {marketplace}")
        if device:
            parts.append(f"Device: {device}")
        return "\n".join(parts)
