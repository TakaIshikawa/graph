"""Adapter for Beeminder datapoints CSV exports."""

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
    parse_float,
    read_csv_rows,
    split_values,
)
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class BeeminderDatapointsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "beeminder_datapoints_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["datapoint", "goal"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types) if entity_types is not None else set(self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        datapoints: list[KnowledgeUnit] = []
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._datapoint_unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                datapoints.append(unit)

        goals = self._goal_units(datapoints)
        if "datapoint" in allowed_types:
            result.units.extend(datapoints)
        if "goal" in allowed_types:
            result.units.extend(goals)
        if {"goal", "datapoint"}.issubset(allowed_types):
            result.edges.extend(self._goal_edges(goals, datapoints))

        result.units.sort(key=lambda unit: (unit.source_entity_type, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _datapoint_unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        goal = first(row, "Goal", "Goal Name")
        goal_slug = first(row, "Goal Slug", "Slug")
        date = self._date(row)
        timestamp = self._timestamp(row)
        created_at = self._parse_datetime(first(row, "Created At", "Created"))
        updated_at = self._parse_datetime(first(row, "Updated At", "Updated"))
        value_text = first(row, "Value", "Val")
        value = parse_float(value_text)
        comment = first(row, "Comment", "Comments")
        tags = split_values(first(row, "Tags", "Tag"))
        metadata = clean_metadata(
            {
                "datapoint_id": first(row, "Datapoint Id", "Datapoint ID", "ID", "Id"),
                "goal": goal,
                "goal_slug": goal_slug,
                "date": date.isoformat() if date else "",
                "timestamp": timestamp.isoformat() if timestamp else "",
                "value": value,
                "value_text": value_text if value is None else "",
                "comment": comment,
                "request_id": first(row, "Request ID", "Request Id", "RequestID"),
                "updated_at": updated_at.isoformat() if updated_at else "",
                "created_at": created_at.isoformat() if created_at else "",
                "daystamp": first(row, "Daystamp", "Day Stamp"),
                "tags": tags,
                "source_file": source_file,
            }
        )
        if not any([metadata.get("datapoint_id"), goal, goal_slug, date, timestamp, value_text, comment, tags]):
            return None

        now = datetime.now(timezone.utc)
        unit_created_at = created_at or timestamp or date or updated_at or now
        unit_updated_at = updated_at or timestamp or date or unit_created_at
        title_goal = goal or goal_slug or "Beeminder goal"
        title_value = value_text or "datapoint"
        title_date = (timestamp or date).date().isoformat() if (timestamp or date) else ""
        title = " ".join(part for part in (title_goal, title_value, title_date) if part)
        return KnowledgeUnit(
            source_project="beeminder_datapoints_csv",
            source_id=self._datapoint_source_id(metadata, goal, goal_slug, date, timestamp, value_text, comment, source_file, index),
            source_entity_type="datapoint",
            title=title,
            content=self._datapoint_content(title, metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["beeminder", "datapoint", goal, goal_slug, *tags] if tag)),
            created_at=unit_created_at,
            updated_at=unit_updated_at,
        )

    def _datapoint_source_id(
        self,
        metadata: dict[str, Any],
        goal: str,
        goal_slug: str,
        date: datetime | None,
        timestamp: datetime | None,
        value_text: str,
        comment: str,
        source_file: str,
        index: int,
    ) -> str:
        if metadata.get("datapoint_id"):
            return digest_source_id("beeminder_datapoints_csv", metadata["datapoint_id"])
        if goal or goal_slug or date or timestamp or value_text or comment:
            return digest_source_id(
                "beeminder_datapoints_csv",
                goal_slug or goal,
                (timestamp or date).isoformat() if (timestamp or date) else "",
                value_text,
                comment,
            )
        return digest_source_id("beeminder_datapoints_csv", source_file, index)

    def _goal_units(self, datapoints: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        slugs: dict[str, str] = {}
        for datapoint in datapoints:
            slug = str(datapoint.metadata.get("goal_slug") or "").strip()
            goal = str(datapoint.metadata.get("goal") or "").strip()
            if not slug and not goal:
                continue
            key = (slug or goal).casefold()
            names.setdefault(key, goal or slug)
            slugs.setdefault(key, slug)
            grouped.setdefault(key, []).append(datapoint)

        units: list[KnowledgeUnit] = []
        for key, goal_datapoints in sorted(grouped.items()):
            name = names[key]
            metadata = self._goal_metadata(goal_datapoints, {"goal": name, "goal_slug": slugs.get(key, "")})
            units.append(
                KnowledgeUnit(
                    source_project="beeminder_datapoints_csv",
                    source_id=digest_source_id("beeminder_datapoints_csv_goal", key),
                    source_entity_type="goal",
                    title=name,
                    content=f"Beeminder goal: {name}\nDatapoints: {len(goal_datapoints)}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=list(dict.fromkeys(tag for tag in ["beeminder", "goal", name, slugs.get(key, "")] if tag)),
                    created_at=min(datapoint.created_at for datapoint in goal_datapoints),
                    updated_at=max(datapoint.updated_at for datapoint in goal_datapoints),
                )
            )
        return units

    def _goal_metadata(self, datapoints: list[KnowledgeUnit], base: dict[str, Any]) -> dict[str, Any]:
        numeric_values = [value for datapoint in datapoints if (value := datapoint.metadata.get("value")) is not None]
        dates = [self._metadata_datetime(datapoint) for datapoint in datapoints]
        dates = [date for date in dates if date is not None]
        metadata = {
            **base,
            "datapoint_count": len(datapoints),
            "value_total": sum(numeric_values) if numeric_values else None,
            "value_minimum": min(numeric_values) if numeric_values else None,
            "value_maximum": max(numeric_values) if numeric_values else None,
            "first_datapoint_at": min(dates).isoformat() if dates else None,
            "last_datapoint_at": max(dates).isoformat() if dates else None,
            "tags": sorted({tag for datapoint in datapoints for tag in datapoint.metadata.get("tags", [])}),
            "request_ids": sorted({str(datapoint.metadata.get("request_id")) for datapoint in datapoints if datapoint.metadata.get("request_id")}),
            "datapoint_source_ids": sorted(datapoint.source_id for datapoint in datapoints),
        }
        return clean_metadata(metadata)

    def _goal_edges(self, goals: list[KnowledgeUnit], datapoints: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        goal_ids = {}
        for goal in goals:
            for key in (goal.metadata.get("goal_slug"), goal.metadata.get("goal")):
                if key:
                    goal_ids[str(key).casefold()] = goal.source_id
        edges: list[KnowledgeEdge] = []
        for datapoint in datapoints:
            key = str(datapoint.metadata.get("goal_slug") or datapoint.metadata.get("goal") or "").casefold()
            goal_id = goal_ids.get(key)
            if not goal_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=digest_source_id("beeminder_datapoints_csv_goal_edge", goal_id, datapoint.source_id),
                    from_unit_id=goal_id,
                    to_unit_id=datapoint.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={"relation_type": "goal_contains_datapoint", "goal": datapoint.metadata.get("goal")},
                )
            )
        return edges

    def _date(self, row: dict[str, Any]) -> datetime | None:
        return self._parse_datetime(first(row, "Date", "Day", "Daystamp"))

    def _timestamp(self, row: dict[str, Any]) -> datetime | None:
        value = first(row, "Timestamp", "Time")
        if not value:
            return None
        parsed_number = parse_float(value)
        if parsed_number is not None and str(value).strip().replace(".", "", 1).isdigit():
            return datetime.fromtimestamp(parsed_number, tz=timezone.utc)
        return self._parse_datetime(value)

    def _parse_datetime(self, value: Any) -> datetime | None:
        parsed = parse_datetime(value)
        if parsed:
            return parsed
        text = "" if value is None else str(value).strip()
        for fmt in ("%Y%m%d",):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _metadata_datetime(self, datapoint: KnowledgeUnit) -> datetime | None:
        for key in ("timestamp", "date", "updated_at", "created_at"):
            parsed = self._parse_datetime(datapoint.metadata.get(key))
            if parsed:
                return parsed
        return None

    def _datapoint_content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (
            ("goal", "Goal"),
            ("goal_slug", "Goal slug"),
            ("date", "Date"),
            ("timestamp", "Timestamp"),
            ("daystamp", "Daystamp"),
            ("value", "Value"),
            ("value_text", "Value"),
            ("comment", "Comment"),
            ("request_id", "Request ID"),
            ("tags", "Tags"),
            ("created_at", "Created"),
            ("updated_at", "Updated"),
        ):
            if metadata.get(key) not in ("", None, []):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
