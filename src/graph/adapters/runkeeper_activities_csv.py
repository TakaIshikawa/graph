"""Adapter for Runkeeper activities CSV exports."""

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
)
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class RunkeeperActivitiesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "runkeeper_activities_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["activity", "route", "activity_type"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types) if entity_types is not None else set(self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        activities: list[KnowledgeUnit] = []
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._activity_unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                activities.append(unit)

        routes = self._route_units(activities)
        activity_types = self._activity_type_units(activities)
        if "activity" in allowed_types:
            result.units.extend(activities)
        if "route" in allowed_types:
            result.units.extend(routes)
        if "activity_type" in allowed_types:
            result.units.extend(activity_types)
        if {"route", "activity"}.issubset(allowed_types):
            result.edges.extend(self._route_edges(routes, activities))
        if {"activity_type", "activity"}.issubset(allowed_types):
            result.edges.extend(self._activity_type_edges(activity_types, activities))

        result.units.sort(key=lambda unit: (unit.source_entity_type, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _activity_unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        activity_id = first(row, "Activity Id", "Activity ID", "ID", "Id")
        activity_type = first(row, "Type", "Activity Type")
        date = self._parse_datetime(first(row, "Date", "Start Time", "Start Date"))
        distance = parse_float(first(row, "Distance", "Distance (mi)", "Distance (km)"))
        duration_seconds = parse_duration_seconds(first(row, "Duration", "Elapsed Time", "Time"))
        route_name = first(row, "Route Name", "Route")
        notes = first(row, "Notes", "Note")
        metadata = clean_metadata(
            {
                "activity_id": activity_id,
                "activity_type": activity_type,
                "date": date.isoformat() if date else "",
                "distance": distance,
                "distance_unit": self._distance_unit(row),
                "duration_seconds": duration_seconds,
                "average_pace": first(row, "Average Pace", "Avg Pace", "Pace"),
                "average_pace_seconds": parse_duration_seconds(first(row, "Average Pace", "Avg Pace", "Pace")),
                "calories": parse_float(first(row, "Calories Burned", "Calories")),
                "climb": parse_float(first(row, "Climb", "Climb (ft)", "Climb (m)", "Elevation Gain", "Total Climb")),
                "climb_unit": self._climb_unit(row),
                "average_heart_rate": parse_float(first(row, "Average Heart Rate", "Avg Heart Rate")),
                "notes": notes,
                "gpx_file": first(row, "GPX File", "GPX", "GPX Filename"),
                "route_name": route_name,
                "source_file": source_file,
            }
        )
        if not any([activity_id, activity_type, date, distance is not None, duration_seconds is not None, route_name, notes]):
            return None

        now = datetime.now(timezone.utc)
        created_at = date or now
        title = " ".join(part for part in (activity_type, date.date().isoformat() if date else "") if part) or "Runkeeper activity"
        return KnowledgeUnit(
            source_project="runkeeper_activities_csv",
            source_id=self._activity_source_id(activity_id, date, activity_type, distance, source_file, index),
            source_entity_type="activity",
            title=title,
            content=self._activity_content(title, metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["runkeeper", "activity", activity_type, route_name] if tag)),
            created_at=created_at,
            updated_at=created_at,
        )

    def _activity_source_id(
        self,
        activity_id: str,
        date: datetime | None,
        activity_type: str,
        distance: float | None,
        source_file: str,
        index: int,
    ) -> str:
        if activity_id:
            return digest_source_id("runkeeper_activities_csv", activity_id)
        if date or activity_type or distance is not None:
            return digest_source_id(
                "runkeeper_activities_csv",
                date.isoformat() if date else "",
                activity_type,
                distance,
            )
        return digest_source_id("runkeeper_activities_csv", source_file, index)

    def _route_units(self, activities: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for activity in activities:
            route = str(activity.metadata.get("route_name") or "").strip()
            if not route:
                continue
            key = route.casefold()
            names.setdefault(key, route)
            grouped.setdefault(key, []).append(activity)

        units: list[KnowledgeUnit] = []
        for key, route_activities in sorted(grouped.items()):
            name = names[key]
            units.append(
                KnowledgeUnit(
                    source_project="runkeeper_activities_csv",
                    source_id=digest_source_id("runkeeper_activities_csv_route", key),
                    source_entity_type="route",
                    title=name,
                    content=f"Runkeeper route: {name}\nActivities: {len(route_activities)}",
                    content_type=ContentType.METADATA,
                    metadata=self._aggregate_metadata(route_activities, {"route_name": name}),
                    tags=["runkeeper", "route", name],
                    created_at=min(activity.created_at for activity in route_activities),
                    updated_at=max(activity.updated_at for activity in route_activities),
                )
            )
        return units

    def _activity_type_units(self, activities: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for activity in activities:
            activity_type = str(activity.metadata.get("activity_type") or "").strip()
            if not activity_type:
                continue
            key = activity_type.casefold()
            names.setdefault(key, activity_type)
            grouped.setdefault(key, []).append(activity)

        units: list[KnowledgeUnit] = []
        for key, type_activities in sorted(grouped.items()):
            name = names[key]
            units.append(
                KnowledgeUnit(
                    source_project="runkeeper_activities_csv",
                    source_id=digest_source_id("runkeeper_activities_csv_activity_type", key),
                    source_entity_type="activity_type",
                    title=name,
                    content=f"Runkeeper activity type: {name}\nActivities: {len(type_activities)}",
                    content_type=ContentType.METADATA,
                    metadata=self._aggregate_metadata(type_activities, {"activity_type": name}),
                    tags=["runkeeper", "activity_type", name],
                    created_at=min(activity.created_at for activity in type_activities),
                    updated_at=max(activity.updated_at for activity in type_activities),
                )
            )
        return units

    def _aggregate_metadata(self, activities: list[KnowledgeUnit], base: dict[str, Any]) -> dict[str, Any]:
        distances = [value for activity in activities if (value := activity.metadata.get("distance")) is not None]
        durations = [value for activity in activities if (value := activity.metadata.get("duration_seconds")) is not None]
        calories = [value for activity in activities if (value := activity.metadata.get("calories")) is not None]
        climbs = [value for activity in activities if (value := activity.metadata.get("climb")) is not None]
        metadata = {
            **base,
            "activity_count": len(activities),
            "total_distance": sum(distances) if distances else None,
            "total_duration_seconds": sum(durations) if durations else None,
            "total_calories": sum(calories) if calories else None,
            "total_climb": sum(climbs) if climbs else None,
            "activity_types": sorted({str(activity.metadata.get("activity_type")) for activity in activities if activity.metadata.get("activity_type")}),
            "routes": sorted({str(activity.metadata.get("route_name")) for activity in activities if activity.metadata.get("route_name")}),
            "activity_source_ids": sorted(activity.source_id for activity in activities),
            "first_activity_at": min(activity.created_at for activity in activities).isoformat(),
            "last_activity_at": max(activity.updated_at for activity in activities).isoformat(),
        }
        return clean_metadata(metadata)

    def _route_edges(self, routes: list[KnowledgeUnit], activities: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        route_ids = {str(route.metadata.get("route_name") or "").casefold(): route.source_id for route in routes}
        edges: list[KnowledgeEdge] = []
        for activity in activities:
            route_id = route_ids.get(str(activity.metadata.get("route_name") or "").casefold())
            if not route_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=digest_source_id("runkeeper_activities_csv_route_edge", route_id, activity.source_id),
                    from_unit_id=route_id,
                    to_unit_id=activity.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={"relation_type": "route_contains_activity", "route_name": activity.metadata.get("route_name")},
                )
            )
        return edges

    def _activity_type_edges(self, activity_types: list[KnowledgeUnit], activities: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        type_ids = {str(unit.metadata.get("activity_type") or "").casefold(): unit.source_id for unit in activity_types}
        edges: list[KnowledgeEdge] = []
        for activity in activities:
            type_id = type_ids.get(str(activity.metadata.get("activity_type") or "").casefold())
            if not type_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=digest_source_id("runkeeper_activities_csv_activity_type_edge", type_id, activity.source_id),
                    from_unit_id=type_id,
                    to_unit_id=activity.source_id,
                    relation=EdgeRelation.RELATES_TO,
                    source=EdgeSource.SOURCE,
                    metadata={"relation_type": "activity_type_activity", "activity_type": activity.metadata.get("activity_type")},
                )
            )
        return edges

    def _parse_datetime(self, value: Any) -> datetime | None:
        parsed = parse_datetime(value)
        if parsed:
            return parsed
        text = "" if value is None else str(value).strip()
        for fmt in ("%m/%d/%Y %I:%M %p", "%m/%d/%Y %I:%M:%S %p", "%b %d, %Y %I:%M %p"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _distance_unit(self, row: dict[str, Any]) -> str:
        for key in row:
            lowered = str(key).casefold()
            if "distance" in lowered and "(km)" in lowered:
                return "km"
            if "distance" in lowered and "(mi)" in lowered:
                return "mi"
        return first(row, "Distance Unit", "Unit")

    def _climb_unit(self, row: dict[str, Any]) -> str:
        for key in row:
            lowered = str(key).casefold()
            if ("climb" in lowered or "elevation" in lowered) and "(m)" in lowered:
                return "m"
            if ("climb" in lowered or "elevation" in lowered) and "(ft)" in lowered:
                return "ft"
        return first(row, "Climb Unit", "Elevation Unit")

    def _activity_content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (
            ("date", "Date"),
            ("activity_type", "Type"),
            ("distance", "Distance"),
            ("duration_seconds", "Duration seconds"),
            ("average_pace", "Average pace"),
            ("calories", "Calories"),
            ("climb", "Climb"),
            ("average_heart_rate", "Average heart rate"),
            ("route_name", "Route"),
            ("gpx_file", "GPX file"),
            ("notes", "Notes"),
        ):
            if metadata.get(key) not in ("", None, []):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
