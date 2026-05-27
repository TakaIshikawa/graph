"""Adapter for Pocket Casts queue CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, parse_duration_seconds, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class PocketCastsQueueCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pocket_casts_queue_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["queued_episode"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "queued_episode" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        episode = first(row, "Episode Title", "Episode", "Title")
        if not episode:
            return None
        podcast = first(row, "Podcast Title", "Podcast")
        url = first(row, "Episode URL", "URL", "Url")
        published_text = first(row, "Published Date", "Published")
        added_text = first(row, "Added Date", "Added")
        published_at = parse_datetime(published_text)
        added_at = parse_datetime(added_text)
        duration = parse_duration_seconds(first(row, "Duration"))
        position = parse_duration_seconds(first(row, "Playback Position", "Position"))
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "podcast_title": podcast,
                "episode_title": episode,
                "url": url,
                "published_date": published_at.isoformat() if published_at else published_text,
                "added_date": added_at.isoformat() if added_at else added_text,
                "duration_seconds": duration,
                "playback_position_seconds": position,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="pocket_casts_queue_csv",
            source_id=digest_source_id("pocket_casts_queue_csv", url or podcast, episode, "" if url else index),
            source_entity_type="queued_episode",
            title=episode,
            content="\n".join(part for part in [episode, f"Podcast: {podcast}" if podcast else "", f"URL: {url}" if url else ""] if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            created_at=published_at or added_at or now,
            updated_at=added_at or published_at or now,
        )
