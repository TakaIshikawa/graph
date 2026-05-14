"""Adapter for Netflix viewing activity CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime
from typing import Any

from graph.adapters._personal_exports import (
    clean_metadata,
    digest_source_id,
    ensure_utc,
    first,
    iter_paths,
    parse_datetime,
    parse_duration_seconds,
    read_csv_rows,
)
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class NetflixViewingActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "netflix_viewing_activity_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["watch_activity"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else set(self.entity_types)
        if "watch_activity" not in allowed:
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.updated_at, unit.source_id)))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = first(row, "Title", "title", "Name", "Video Title")
        if not title:
            return None
        watched_at = parse_datetime(first(row, "Date", "Watched Date", "Watched At", "Start Time", "StartTime"))
        if watched_at is None:
            return None

        profile = first(row, "Profile Name", "Profile", "profile_name", "profile")
        duration = first(row, "Duration", "duration")
        device = first(row, "Device Type", "Device", "device_type", "device")
        country = first(row, "Country", "country")
        metadata = clean_metadata(
            {
                "title": title,
                "profile_name": profile,
                "watched_at": watched_at.isoformat(),
                "watched_date": watched_at.date().isoformat(),
                "duration": duration,
                "duration_seconds": parse_duration_seconds(duration),
                "device": device,
                "country": country,
                "attributes": first(row, "Attributes", "attributes"),
                "supplemental_video_type": first(row, "Supplemental Video Type", "supplemental_video_type"),
                "bookmark": first(row, "Bookmark", "bookmark"),
                "latest_bookmark": first(row, "Latest Bookmark", "latest_bookmark"),
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.NETFLIX_VIEWING_ACTIVITY_CSV,
            source_id=digest_source_id("netflix_viewing_activity_csv", title, profile, watched_at.isoformat()),
            source_entity_type="watch_activity",
            title=title,
            content=self._content(title, profile, watched_at, duration, device, country),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["netflix", "watch_activity"],
            created_at=watched_at,
            updated_at=watched_at,
        )

    def _content(self, title: str, profile: str, watched_at: datetime, duration: str, device: str, country: str) -> str:
        parts = [f"Watched: {title}", f"Watched at: {watched_at.isoformat()}"]
        if profile:
            parts.append(f"Profile: {profile}")
        if duration:
            parts.append(f"Duration: {duration}")
        if device:
            parts.append(f"Device: {device}")
        if country:
            parts.append(f"Country: {country}")
        return "\n".join(parts)
