"""Adapter for Apple Health export.xml workout records."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class AppleHealthWorkoutsAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "apple_health_workouts"

    @property
    def entity_types(self) -> list[str]:
        return ["workout", "monthly_aggregate"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = self._sync_datetime(since) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                for element in self._iter_workouts(path):
                    unit = self._unit_from_workout(element, path.name)
                    if unit is None:
                        continue
                    if sync_at and unit.created_at <= sync_at:
                        continue
                    units.append(unit)
                    element.clear()
            except (OSError, ET.ParseError, UnicodeDecodeError):
                continue

        workouts = sorted(units, key=lambda unit: (unit.created_at, unit.source_id))
        aggregates = self._monthly_aggregate_units(workouts) if "monthly_aggregate" in allowed_types else []
        if "monthly_aggregate" in allowed_types:
            result.units.extend(aggregates)
        if "workout" in allowed_types:
            result.units.extend(workouts)
        if {"workout", "monthly_aggregate"}.issubset(allowed_types):
            result.edges.extend(self._monthly_aggregate_edges(workouts, aggregates))
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []

        root = Path(self.path).expanduser()
        if root.is_file():
            return [root] if root.name == "export.xml" or root.suffix.lower() == ".xml" else []
        if not root.is_dir():
            return []

        export = root / "export.xml"
        if export.is_file():
            return [export]
        return sorted(
            (child for child in root.iterdir() if child.is_file() and child.suffix.lower() == ".xml"),
            key=lambda child: child.name,
        )

    def _iter_workouts(self, path: Path) -> Any:
        for _event, element in ET.iterparse(path, events=("end",)):
            if self._local_name(element.tag) == "Workout":
                yield element

    def _unit_from_workout(self, workout: ET.Element, source_file: str) -> KnowledgeUnit | None:
        start_at = self._parse_datetime(workout.attrib.get("startDate"))
        end_at = self._parse_datetime(workout.attrib.get("endDate"))
        created_at = start_at or end_at or self._parse_datetime(workout.attrib.get("creationDate"))
        if created_at is None:
            return None

        raw_activity_type = self._text(workout.attrib.get("workoutActivityType"))
        activity_type = self._activity_type(raw_activity_type)
        duration = self._parse_float(workout.attrib.get("duration"))
        distance = self._parse_float(workout.attrib.get("totalDistance"))
        calories = self._parse_float(workout.attrib.get("totalEnergyBurned"))
        metadata_entries = self._metadata_entries(workout)
        routes = self._route_metadata(workout)

        metadata = {
            "activity_type": activity_type,
            "workout_activity_type": raw_activity_type,
            "duration": duration,
            "duration_unit": self._text(workout.attrib.get("durationUnit")),
            "distance": distance,
            "distance_unit": self._text(workout.attrib.get("totalDistanceUnit")),
            "calories": calories,
            "calories_unit": self._text(workout.attrib.get("totalEnergyBurnedUnit")),
            "start_at": start_at.isoformat() if start_at else None,
            "end_at": end_at.isoformat() if end_at else None,
            "creation_date": self._datetime_metadata(workout.attrib.get("creationDate")),
            "source_name": self._text(workout.attrib.get("sourceName")),
            "source_version": self._text(workout.attrib.get("sourceVersion")),
            "device": self._text(workout.attrib.get("device")),
            "metadata_entries": metadata_entries,
            "routes": routes,
            "source_file": source_file,
        }

        return KnowledgeUnit(
            source_project=SourceProject.APPLE_HEALTH_WORKOUTS,
            source_id=self._source_id(workout, created_at),
            source_entity_type="workout",
            title=self._title(activity_type, start_at),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["apple_health", "workout", activity_type.lower()] if activity_type else ["apple_health", "workout"],
            created_at=created_at,
            updated_at=end_at or created_at,
        )

    def _metadata_entries(self, workout: ET.Element) -> dict[str, str]:
        entries: dict[str, str] = {}
        for child in workout:
            if self._local_name(child.tag) != "MetadataEntry":
                continue
            key = self._text(child.attrib.get("key"))
            if not key:
                continue
            entries[key] = self._text(child.attrib.get("value"))
        return entries

    def _route_metadata(self, workout: ET.Element) -> list[dict[str, str]]:
        routes: list[dict[str, str]] = []
        for child in workout:
            local_name = self._local_name(child.tag)
            if "Route" not in local_name and "route" not in local_name.lower():
                continue
            route = {key: self._text(value) for key, value in child.attrib.items()}
            if route:
                route["element"] = local_name
                routes.append(route)
        return routes

    def _source_id(self, workout: ET.Element, created_at: datetime) -> str:
        metadata_entries = self._metadata_entries(workout)
        external_id = (
            metadata_entries.get("HKMetadataKeyExternalUUID")
            or metadata_entries.get("HKMetadataKeySyncIdentifier")
            or metadata_entries.get("externalUUID")
        )
        if external_id:
            identifier = external_id
        else:
            identifier = "|".join(
                [
                    created_at.isoformat(),
                    self._text(workout.attrib.get("endDate")),
                    self._text(workout.attrib.get("workoutActivityType")),
                    self._text(workout.attrib.get("sourceName")),
                    self._text(workout.attrib.get("duration")),
                    self._text(workout.attrib.get("totalDistance")),
                    self._text(workout.attrib.get("totalEnergyBurned")),
                ]
            )
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:24]
        return f"apple_health_workouts:{digest}"

    def _monthly_aggregate_units(self, workouts: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[tuple[str, str], list[KnowledgeUnit]] = {}
        for workout in workouts:
            month = workout.created_at.strftime("%Y-%m")
            activity_type = str(workout.metadata.get("activity_type") or "").strip()
            if not activity_type:
                continue
            grouped.setdefault((month, activity_type.casefold()), []).append(workout)

        units: list[KnowledgeUnit] = []
        for (month, _normalized_type), items in grouped.items():
            activity_type = str(items[0].metadata.get("activity_type"))
            total_duration = sum(float(item.metadata.get("duration") or 0) for item in items if item.metadata.get("duration") is not None)
            total_distance = sum(float(item.metadata.get("distance") or 0) for item in items if item.metadata.get("distance") is not None)
            metadata = {
                "month": month,
                "activity_type": activity_type,
                "workout_count": len(items),
                "total_duration": total_duration if total_duration else None,
                "duration_unit": items[0].metadata.get("duration_unit"),
                "total_distance": total_distance if total_distance else None,
                "distance_unit": items[0].metadata.get("distance_unit"),
                "workout_source_ids": [item.source_id for item in items],
            }
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.APPLE_HEALTH_WORKOUTS,
                    source_id=self._monthly_aggregate_source_id(month, activity_type),
                    source_entity_type="monthly_aggregate",
                    title=f"{activity_type} workouts in {month}",
                    content=f"Apple Health {activity_type} workouts in {month}",
                    content_type=ContentType.METADATA,
                    metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
                    tags=["apple_health", "workout", "monthly_aggregate", activity_type.lower()],
                    created_at=min(item.created_at for item in items),
                    updated_at=max(item.updated_at for item in items),
                )
            )
        return units

    def _monthly_aggregate_edges(self, workouts: list[KnowledgeUnit], aggregates: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        aggregate_ids = {
            (unit.metadata.get("month"), str(unit.metadata.get("activity_type") or "").casefold()): unit.source_id
            for unit in aggregates
        }
        edges: list[KnowledgeEdge] = []
        for workout in workouts:
            key = (workout.created_at.strftime("%Y-%m"), str(workout.metadata.get("activity_type") or "").casefold())
            target = aggregate_ids.get(key)
            if target:
                edges.append(self._edge(workout.source_id, target, "workout_monthly_aggregate"))
        return list({edge.id: edge for edge in edges}.values())

    def _monthly_aggregate_source_id(self, month: str, activity_type: str) -> str:
        digest = hashlib.sha256(f"{month}|{activity_type.casefold()}".encode("utf-8")).hexdigest()[:24]
        return f"apple_health_workouts:monthly_aggregate:{digest}"

    def _edge(self, from_id: str, to_id: str, relation_type: str) -> KnowledgeEdge:
        digest = hashlib.sha256(f"{from_id}|{relation_type}|{to_id}".encode("utf-8")).hexdigest()[:24]
        return KnowledgeEdge(
            id=f"apple_health_workouts:edge:{digest}",
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.SOURCE,
            metadata={"source_project": SourceProject.APPLE_HEALTH_WORKOUTS.value, "relation_type": relation_type},
        )

    def _title(self, activity_type: str, start_at: datetime | None) -> str:
        if start_at:
            return f"{activity_type or 'Workout'} on {start_at.date().isoformat()}"
        return activity_type or "Apple Health workout"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = []
        if metadata["activity_type"]:
            parts.append(f"Activity: {metadata['activity_type']}")
        if metadata["duration"] is not None:
            parts.append(f"Duration: {metadata['duration']} {metadata['duration_unit']}".strip())
        if metadata["distance"] is not None:
            parts.append(f"Distance: {metadata['distance']} {metadata['distance_unit']}".strip())
        if metadata["calories"] is not None:
            parts.append(f"Calories: {metadata['calories']} {metadata['calories_unit']}".strip())
        if metadata["start_at"]:
            parts.append(f"Start: {metadata['start_at']}")
        if metadata["end_at"]:
            parts.append(f"End: {metadata['end_at']}")
        if metadata["source_name"]:
            parts.append(f"Source: {metadata['source_name']}")
        return "\n".join(parts)

    def _activity_type(self, value: str) -> str:
        if not value:
            return ""
        return value.removeprefix("HKWorkoutActivityType")

    def _parse_float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(str(value).strip())
        except ValueError:
            return None

    def _datetime_metadata(self, value: Any) -> str | None:
        parsed = self._parse_datetime(value)
        return parsed.isoformat() if parsed else None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return self._ensure_utc(value)

        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
        return self._ensure_utc(parsed)

    def _sync_datetime(self, since: SyncState) -> datetime:
        return self._ensure_utc(since.last_sync_at)

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _local_name(self, tag: str) -> str:
        return tag.rsplit("}", 1)[-1]
