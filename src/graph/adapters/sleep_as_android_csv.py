"""Adapter for Sleep as Android CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import (
    clean_metadata,
    digest_source_id,
    ensure_utc,
    first,
    iter_paths,
    parse_datetime,
    parse_duration_seconds,
    parse_float,
    read_csv_rows,
    split_values,
)
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class SleepAsAndroidCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "sleep_as_android_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["sleep_session"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else set(self.entity_types)
        if "sleep_session" not in allowed:
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
        start = parse_datetime(first(row, "Sleep Start", "Start", "From", "From Time", "Start Time", "start"))
        end = parse_datetime(first(row, "Sleep End", "End", "To", "To Time", "End Time", "end"))
        updated = parse_datetime(first(row, "Updated", "Updated At", "Modified", "Last Modified"))
        duration = first(row, "Duration", "Length", "Sleep Duration")
        duration_seconds = parse_duration_seconds(duration)
        if start is None and not duration:
            return None
        event_at = updated or start or end or datetime.now(timezone.utc)
        quality = parse_float(first(row, "Quality", "Rating", "Sleep Quality", "Score"))
        deep_sleep = parse_float(first(row, "Deep Sleep", "Deep Sleep %", "DeepSleep", "Deep Sleep Percentage"))
        snoring = parse_float(first(row, "Snoring", "Snoring Time", "Snore", "Snore Time"))
        noise = parse_float(first(row, "Noise", "Noise Level", "Avg Noise", "Average Noise"))
        tags = split_values(first(row, "Tags", "Labels", "Tag"))
        comments = first(row, "Comment", "Comments", "Notes", "Note")
        metadata = clean_metadata(
            {
                "sleep_start": start.isoformat() if start else None,
                "sleep_end": end.isoformat() if end else None,
                "duration": duration,
                "duration_seconds": duration_seconds,
                "quality": quality,
                "rating": quality,
                "deep_sleep": deep_sleep,
                "snoring": snoring,
                "noise": noise,
                "tags": tags,
                "comments": comments,
                "updated_at": updated.isoformat() if updated else None,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.SLEEP_AS_ANDROID_CSV,
            source_id=digest_source_id("sleep_as_android_csv", start, end, duration, comments),
            source_entity_type="sleep_session",
            title=f"Sleep session {(start or event_at).date().isoformat()}",
            content=self._content(start, end, duration, quality, deep_sleep, tags, comments),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["sleep_as_android", "sleep_session", *tags],
            created_at=start or event_at,
            updated_at=event_at,
        )

    def _content(
        self,
        start: datetime | None,
        end: datetime | None,
        duration: str,
        quality: float | None,
        deep_sleep: float | None,
        tags: list[str],
        comments: str,
    ) -> str:
        parts: list[str] = []
        if start:
            parts.append(f"Start: {start.isoformat()}")
        if end:
            parts.append(f"End: {end.isoformat()}")
        if duration:
            parts.append(f"Duration: {duration}")
        if quality is not None:
            parts.append(f"Quality: {quality}")
        if deep_sleep is not None:
            parts.append(f"Deep sleep: {deep_sleep}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if comments:
            parts.append(comments)
        return "\n".join(parts)
