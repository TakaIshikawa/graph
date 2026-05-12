"""Adapter for Garmin Connect activities CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_duration_seconds, parse_float, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class GarminActivitiesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "garmin_activities_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["activity", "activity_type"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        activities: list[KnowledgeUnit] = []
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                activities.append(unit)
                if "activity" in allowed_types:
                    result.units.append(unit)
        activity_types = self._activity_type_units(activities) if "activity_type" in allowed_types else []
        if "activity_type" in allowed_types:
            result.units.extend(activity_types)
        if {"activity", "activity_type"}.issubset(allowed_types):
            result.edges.extend(self._activity_type_edges(activities, activity_types))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        activity_id = first(row, "Activity ID", "ID")
        title = first(row, "Title", "Name")
        sport = first(row, "Activity Type", "Type")
        if not activity_id and not title and not sport:
            return None
        date = parse_datetime(first(row, "Date", "Start Time", "Start"))
        metadata = {
            "activity_id": activity_id,
            "activity_type": sport,
            "favorite": first(row, "Favorite"),
            "title": title,
            "date": date.isoformat() if date else first(row, "Date"),
            "distance": parse_float(first(row, "Distance")),
            "calories": parse_int(first(row, "Calories")),
            "duration_seconds": parse_duration_seconds(first(row, "Time", "Duration")),
            "avg_hr": parse_int(first(row, "Avg HR", "Average HR")),
            "max_hr": parse_int(first(row, "Max HR")),
            "aerobic_te": parse_float(first(row, "Aerobic TE")),
            "avg_speed": parse_float(first(row, "Avg Speed")),
            "max_speed": parse_float(first(row, "Max Speed")),
            "total_ascent": parse_float(first(row, "Total Ascent")),
            "total_descent": parse_float(first(row, "Total Descent")),
            "source_file": source_file,
        }
        now = datetime.now(timezone.utc)
        name = title or sport or f"Garmin activity {activity_id}"
        return KnowledgeUnit(
            source_project=SourceProject.GARMIN_ACTIVITIES_CSV,
            source_id=f"garmin_activities_csv:{activity_id}" if activity_id else digest_source_id("garmin_activities_csv", name, date),
            source_entity_type="activity",
            title=name,
            content=self._content(name, metadata),
            content_type=ContentType.METADATA,
            metadata=clean_metadata(metadata),
            tags=list(dict.fromkeys(tag for tag in ["garmin", "activity", sport] if tag)),
            created_at=date or now,
            updated_at=date or now,
        )

    def _content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (("activity_type", "Type"), ("distance", "Distance"), ("duration_seconds", "Duration seconds")):
            if metadata.get(key) not in ("", None):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)

    def _activity_type_units(self, activities: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        labels: dict[str, str] = {}
        for activity in activities:
            label = str(activity.metadata.get("activity_type") or "").strip()
            key = self._normalize_activity_type(label)
            if not key:
                continue
            grouped.setdefault(key, []).append(activity)
            labels.setdefault(key, label)
        units: list[KnowledgeUnit] = []
        for key, items in grouped.items():
            total_distance = sum(float(item.metadata.get("distance") or 0) for item in items if item.metadata.get("distance") is not None)
            total_duration = sum(int(item.metadata.get("duration_seconds") or 0) for item in items if item.metadata.get("duration_seconds") is not None)
            metadata = {
                "activity_type": labels[key],
                "normalized_activity_type": key,
                "activity_count": len(items),
                "total_distance": total_distance if total_distance else None,
                "total_duration_seconds": total_duration if total_duration else None,
                "activity_source_ids": [item.source_id for item in items],
            }
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GARMIN_ACTIVITIES_CSV,
                    source_id=self._activity_type_source_id(key),
                    source_entity_type="activity_type",
                    title=f"Garmin {labels[key]} activities",
                    content=f"Garmin activity type: {labels[key]}",
                    content_type=ContentType.METADATA,
                    metadata=clean_metadata(metadata),
                    tags=["garmin", "activity_type", labels[key]],
                    created_at=min(item.created_at for item in items),
                    updated_at=max(item.updated_at for item in items),
                )
            )
        return units

    def _activity_type_edges(self, units: list[KnowledgeUnit], activity_types: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        type_ids = {unit.metadata["normalized_activity_type"]: unit.source_id for unit in activity_types}
        edges: list[KnowledgeEdge] = []
        for activity in units:
            if activity.source_entity_type != "activity":
                continue
            key = self._normalize_activity_type(activity.metadata.get("activity_type"))
            target = type_ids.get(key)
            if target:
                edges.append(self._edge(activity.source_id, target, "activity_type"))
        return list({edge.id: edge for edge in edges}.values())

    def _edge(self, from_id: str, to_id: str, relation_type: str) -> KnowledgeEdge:
        digest = digest_source_id("garmin_activities_csv:edge", from_id, relation_type, to_id).rsplit(":", 1)[-1]
        return KnowledgeEdge(
            id=f"garmin_activities_csv:edge:{digest}",
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.SOURCE,
            metadata={"source_project": SourceProject.GARMIN_ACTIVITIES_CSV.value, "relation_type": relation_type},
        )

    def _normalize_activity_type(self, value: Any) -> str:
        return " ".join(str(value or "").strip().casefold().split())

    def _activity_type_source_id(self, normalized: str) -> str:
        return digest_source_id("garmin_activities_csv:activity_type", normalized)
