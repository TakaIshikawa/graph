"""Adapter for Fitbit daily activity CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class FitbitDailyActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "fitbit_daily_activity_csv"

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
        date_text = first(row, "Date", "Activity Date", "dateTime", "Day")
        date = parse_datetime(date_text)
        steps = parse_int(first(row, "Steps", "Step Count"))
        distance = parse_float(first(row, "Distance", "Distance (mi)", "Distance (km)"))
        floors = parse_int(first(row, "Floors", "Floors Climbed"))
        calories = parse_int(first(row, "Calories Burned", "Calories", "Calories Out"))
        lightly_active = parse_int(first(row, "Minutes Lightly Active", "Lightly Active Minutes"))
        fairly_active = parse_int(first(row, "Minutes Fairly Active", "Fairly Active Minutes"))
        very_active = parse_int(first(row, "Minutes Very Active", "Very Active Minutes"))
        sedentary = parse_int(first(row, "Sedentary Minutes", "Minutes Sedentary"))
        activity_calories = parse_int(first(row, "Activity Calories", "Calories Activity"))
        if not any([date_text, steps is not None, distance is not None, calories is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "date": date.date().isoformat() if date else date_text,
                "steps": steps,
                "distance": distance,
                "distance_unit": self._distance_unit(row),
                "floors": floors,
                "calories": calories,
                "lightly_active_minutes": lightly_active,
                "fairly_active_minutes": fairly_active,
                "very_active_minutes": very_active,
                "sedentary_minutes": sedentary,
                "activity_calories": activity_calories,
                "goals": self._prefixed_fields(row, "goal") or None,
                "progress": self._prefixed_fields(row, "progress") or None,
                "source_file": source_file,
            }
        )
        created_at = date or now
        return KnowledgeUnit(
            source_project=SourceProject.FITBIT_DAILY_ACTIVITY_CSV,
            source_id=digest_source_id("fitbit_daily_activity_csv", metadata.get("date"), index if not metadata.get("date") else ""),
            source_entity_type="daily_activity",
            title=f"Fitbit activity {metadata.get('date', 'unknown date')}",
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["fitbit", "daily_activity"],
            created_at=created_at,
            updated_at=created_at,
        )

    def _distance_unit(self, row: dict[str, Any]) -> str:
        for key in row:
            lowered = str(key).casefold()
            if "distance" in lowered and "(km)" in lowered:
                return "km"
            if "distance" in lowered and "(mi)" in lowered:
                return "mi"
        return first(row, "Distance Unit", "Unit")

    def _prefixed_fields(self, row: dict[str, Any], prefix: str) -> dict[str, int | float | str]:
        values: dict[str, int | float | str] = {}
        for key, value in row.items():
            text = "" if value is None else str(value).strip()
            normalized = str(key).strip().casefold()
            if not text or prefix not in normalized:
                continue
            parsed_int = parse_int(text)
            parsed_float = parse_float(text)
            clean_key = normalized.replace(" ", "_").replace("/", "_")
            values[clean_key] = parsed_int if parsed_int is not None and parsed_float == parsed_int else parsed_float if parsed_float is not None else text
        return values

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [f"Date: {metadata.get('date')}" if metadata.get("date") else ""]
        for key, label in (
            ("steps", "Steps"),
            ("distance", "Distance"),
            ("calories", "Calories"),
            ("floors", "Floors"),
            ("very_active_minutes", "Very active minutes"),
            ("fairly_active_minutes", "Fairly active minutes"),
            ("lightly_active_minutes", "Lightly active minutes"),
            ("sedentary_minutes", "Sedentary minutes"),
            ("activity_calories", "Activity calories"),
        ):
            if metadata.get(key) is not None:
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(part for part in parts if part)
