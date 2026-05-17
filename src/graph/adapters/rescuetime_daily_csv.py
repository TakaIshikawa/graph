"""Adapter for RescueTime daily summary CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class RescueTimeDailyCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "rescuetime_daily_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["daily_activity"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "daily_activity" not in entity_types:
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
        date_text = first(row, "Date", "Day")
        timestamp = parse_datetime(date_text)
        productivity_pulse = parse_float(first(row, "Productivity Pulse", "Pulse"))
        very_productive = self._metric(row, "Very Productive", "Very Productive Hours")
        productive = self._metric(row, "Productive", "Productive Hours")
        neutral = self._metric(row, "Neutral", "Neutral Hours")
        distracting = self._metric(row, "Distracting", "Distracting Hours")
        very_distracting = self._metric(row, "Very Distracting", "Very Distracting Hours")
        total_hours = parse_float(first(row, "Total Hours", "Hours"))
        total_seconds = parse_int(first(row, "Total Seconds", "Seconds", "Total Time Seconds"))
        category = first(row, "Category")
        activity = first(row, "Activity", "Application", "Website")
        details = first(row, "Details", "Detail", "Description")

        if not any(
            [
                date_text,
                productivity_pulse is not None,
                very_productive is not None,
                productive is not None,
                neutral is not None,
                distracting is not None,
                very_distracting is not None,
                total_hours is not None,
                total_seconds is not None,
                category,
                activity,
                details,
            ]
        ):
            return None

        if total_seconds is None and total_hours is not None:
            total_seconds = int(round(total_hours * 3600))
        if total_hours is None and total_seconds is not None:
            total_hours = round(total_seconds / 3600, 4)

        now = datetime.now(timezone.utc)
        date_value = timestamp.date().isoformat() if timestamp else date_text
        metadata = clean_metadata(
            {
                "date": date_value,
                "productivity_pulse": productivity_pulse,
                "very_productive": very_productive,
                "productive": productive,
                "neutral": neutral,
                "distracting": distracting,
                "very_distracting": very_distracting,
                "total_hours": total_hours,
                "total_seconds": total_seconds,
                "category": category,
                "activity": activity,
                "details": details,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        created_at = timestamp or now
        return KnowledgeUnit(
            source_project="rescuetime_daily_csv",
            source_id=digest_source_id("rescuetime_daily_csv", date_value, category, activity, details, index if not date_value else ""),
            source_entity_type="daily_activity",
            title=self._title(date_value, category, activity),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["rescuetime", "daily_activity", category] if tag)),
            created_at=created_at,
            updated_at=created_at,
        )

    def _metric(self, row: dict[str, Any], *keys: str) -> float | None:
        return parse_float(first(row, *keys))

    def _title(self, date_value: str, category: str, activity: str) -> str:
        scope = activity or category
        if scope:
            return f"RescueTime {date_value or 'unknown date'}: {scope}"
        return f"RescueTime {date_value or 'unknown date'}"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [f"Date: {metadata.get('date')}" if metadata.get("date") else ""]
        for key, label in (
            ("productivity_pulse", "Productivity pulse"),
            ("total_hours", "Total hours"),
            ("total_seconds", "Total seconds"),
            ("very_productive", "Very productive"),
            ("productive", "Productive"),
            ("neutral", "Neutral"),
            ("distracting", "Distracting"),
            ("very_distracting", "Very distracting"),
            ("category", "Category"),
            ("activity", "Activity"),
            ("details", "Details"),
        ):
            if metadata.get(key) is not None:
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(part for part in parts if part)
