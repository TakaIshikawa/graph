"""Adapter for Google Fit daily summary CSV exports."""

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
    parse_int,
    read_csv_rows,
)
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class GoogleFitDailyCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_fit_daily_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["daily_activity", "metric"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types) if entity_types is not None else set(self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        daily_units: list[KnowledgeUnit] = []
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._daily_unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                daily_units.append(unit)

        metric_units = self._metric_units(daily_units)
        if "daily_activity" in allowed_types:
            result.units.extend(daily_units)
        if "metric" in allowed_types:
            result.units.extend(metric_units)
        if {"daily_activity", "metric"}.issubset(allowed_types):
            result.edges.extend(self._metric_edges(metric_units, daily_units))

        result.units.sort(key=lambda unit: (unit.source_entity_type, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _daily_unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        date_text = first(row, "Date", "Activity Date", "Day")
        date = parse_datetime(date_text)
        metadata = clean_metadata(
            {
                "date": date.date().isoformat() if date else date_text,
                "steps": parse_int(first(row, "Steps", "Step Count")),
                "distance": parse_float(first(row, "Distance", "Distance (m)", "Distance (km)", "Distance (mi)")),
                "distance_unit": self._distance_unit(row),
                "move_minutes": parse_int(first(row, "Move Minutes", "Move Min")),
                "heart_points": parse_int(first(row, "Heart Points", "Heart Point")),
                "calories": parse_float(first(row, "Calories", "Calories Burned")),
                "active_calories": parse_float(first(row, "Active Calories", "Active Energy")),
                "average_heart_rate": parse_float(first(row, "Average Heart Rate", "Avg Heart Rate")),
                "min_heart_rate": parse_float(first(row, "Min Heart Rate", "Minimum Heart Rate")),
                "max_heart_rate": parse_float(first(row, "Max Heart Rate", "Maximum Heart Rate")),
                "sleep_duration_seconds": parse_duration_seconds(first(row, "Sleep Duration", "Sleep")),
                "weight": parse_float(first(row, "Weight", "Weight (kg)", "Weight (lb)")),
                "weight_unit": self._weight_unit(row),
                "source": first(row, "Source", "Data Source"),
                "source_file": source_file,
            }
        )
        metric_values = {key: metadata[key] for key in self._metric_keys() if key in metadata}
        if not any([metadata.get("date"), metric_values, metadata.get("source")]):
            return None

        now = datetime.now(timezone.utc)
        created_at = date or now
        return KnowledgeUnit(
            source_project="google_fit_daily_csv",
            source_id=self._daily_source_id(metadata, source_file, index),
            source_entity_type="daily_activity",
            title=f"Google Fit activity {metadata.get('date', 'unknown date')}",
            content=self._daily_content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["google_fit", "daily_activity", metadata.get("source")] if tag)),
            created_at=created_at,
            updated_at=created_at,
        )

    def _daily_source_id(self, metadata: dict[str, Any], source_file: str, index: int) -> str:
        if metadata.get("date"):
            return digest_source_id("google_fit_daily_csv", metadata["date"])
        return digest_source_id("google_fit_daily_csv", source_file, index)

    def _metric_units(self, daily_units: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        units: list[KnowledgeUnit] = []
        for key in self._metric_keys():
            values = [(daily, daily.metadata.get(key)) for daily in daily_units if daily.metadata.get(key) is not None]
            if not values:
                continue
            numeric_values = [float(value) for _, value in values]
            metadata = {
                "metric": key,
                "daily_count": len(values),
                "total": sum(numeric_values),
                "minimum": min(numeric_values),
                "maximum": max(numeric_values),
                "average": sum(numeric_values) / len(numeric_values),
                "daily_source_ids": sorted(daily.source_id for daily, _ in values),
                "first_day": min(daily.created_at for daily, _ in values).date().isoformat(),
                "last_day": max(daily.created_at for daily, _ in values).date().isoformat(),
                "sources": sorted({str(daily.metadata.get("source")) for daily, _ in values if daily.metadata.get("source")}),
            }
            units.append(
                KnowledgeUnit(
                    source_project="google_fit_daily_csv",
                    source_id=digest_source_id("google_fit_daily_csv_metric", key),
                    source_entity_type="metric",
                    title=f"Google Fit {key.replace('_', ' ')}",
                    content=f"Google Fit metric: {key.replace('_', ' ')}\nDays: {len(values)}",
                    content_type=ContentType.METADATA,
                    metadata=clean_metadata(metadata),
                    tags=["google_fit", "metric", key],
                    created_at=min(daily.created_at for daily, _ in values),
                    updated_at=max(daily.updated_at for daily, _ in values),
                )
            )
        return units

    def _metric_edges(self, metric_units: list[KnowledgeUnit], daily_units: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        metric_ids = {str(metric.metadata.get("metric")): metric.source_id for metric in metric_units}
        edges: list[KnowledgeEdge] = []
        for daily in daily_units:
            for key in self._metric_keys():
                if daily.metadata.get(key) is None or key not in metric_ids:
                    continue
                edges.append(
                    KnowledgeEdge(
                        id=digest_source_id("google_fit_daily_csv_metric_edge", metric_ids[key], daily.source_id),
                        from_unit_id=metric_ids[key],
                        to_unit_id=daily.source_id,
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={"relation_type": "metric_contains_daily_activity", "metric": key, "value": daily.metadata.get(key)},
                    )
                )
        return edges

    def _metric_keys(self) -> tuple[str, ...]:
        return (
            "steps",
            "distance",
            "move_minutes",
            "heart_points",
            "calories",
            "active_calories",
            "average_heart_rate",
            "min_heart_rate",
            "max_heart_rate",
            "sleep_duration_seconds",
            "weight",
        )

    def _distance_unit(self, row: dict[str, Any]) -> str:
        for key in row:
            lowered = str(key).casefold()
            if "distance" in lowered and "(km)" in lowered:
                return "km"
            if "distance" in lowered and "(mi)" in lowered:
                return "mi"
            if "distance" in lowered and "(m)" in lowered:
                return "m"
        return first(row, "Distance Unit", "Distance Units", "Unit")

    def _weight_unit(self, row: dict[str, Any]) -> str:
        for key in row:
            lowered = str(key).casefold()
            if "weight" in lowered and "(kg)" in lowered:
                return "kg"
            if "weight" in lowered and "(lb)" in lowered:
                return "lb"
        return first(row, "Weight Unit", "Weight Units")

    def _daily_content(self, metadata: dict[str, Any]) -> str:
        parts = [f"Date: {metadata.get('date')}" if metadata.get("date") else "Google Fit daily activity"]
        for key, label in (
            ("steps", "Steps"),
            ("distance", "Distance"),
            ("move_minutes", "Move minutes"),
            ("heart_points", "Heart points"),
            ("calories", "Calories"),
            ("active_calories", "Active calories"),
            ("average_heart_rate", "Average heart rate"),
            ("min_heart_rate", "Min heart rate"),
            ("max_heart_rate", "Max heart rate"),
            ("sleep_duration_seconds", "Sleep duration seconds"),
            ("weight", "Weight"),
            ("source", "Source"),
        ):
            if metadata.get(key) not in ("", None, []):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
