"""Adapter for Toggl time tracking exports."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class TogglAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "toggl"

    @property
    def entity_types(self) -> list[str]:
        return ["time_entry"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "time_entry" not in entity_types:
            return result

        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists() or not path.is_file():
            return result

        try:
            items = self._read_items(path)
        except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError):
            return result

        sync_at = self._sync_datetime(since) if since else None
        for item in items:
            description = self._first(item, "Description", "description", "task")
            project = self._first(item, "Project", "project", "project_name")
            if not description and not project:
                continue

            # Parse start time - try combining date and time fields first
            start_date = self._first(item, "Start date", "start_date", "date")
            start_time = self._first(item, "Start time", "start_time")
            if start_date and start_time:
                start_datetime_str = f"{start_date} {start_time}"
                start_at = self._parse_datetime_with_time(start_datetime_str)
            else:
                start_at = self._parse_datetime(
                    self._first(item, "Start", "start", "started_at", "start_datetime")
                )

            if sync_at and start_at and start_at <= sync_at:
                continue

            # Parse end time
            end_date = self._first(item, "End date", "end_date")
            end_time = self._first(item, "End time", "end_time")
            if end_date and end_time:
                end_datetime_str = f"{end_date} {end_time}"
                end_at = self._parse_datetime_with_time(end_datetime_str)
            else:
                end_at = self._parse_datetime(self._first(item, "End", "end", "ended_at", "end_datetime"))

            duration = self._first(item, "Duration", "duration")
            tags = self._tags(item)
            billable = self._first(item, "Billable", "billable", "is_billable")

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.TOGGL,
                    source_id=self._source_id(description, project, start_at, duration),
                    source_entity_type="time_entry",
                    title=self._format_title(description, project),
                    content=self._content(description, project, duration, tags, billable),
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "description": description,
                        "project": project,
                        "start": start_date or self._first(item, "Start", "start", "started_at"),
                        "end": end_date or self._first(item, "End", "end", "ended_at"),
                        "duration": duration,
                        "tags": tags,
                        "billable": billable,
                    },
                    tags=tags,
                    created_at=start_at or datetime.now(timezone.utc),
                    updated_at=end_at or start_at or datetime.now(timezone.utc),
                )
            )

        return result

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                return [
                    {str(key).strip(): value for key, value in row.items() if key is not None}
                    for row in csv.DictReader(handle)
                ]

        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._json_items(parsed)

    def _json_items(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []

        for key in ("time_entries", "entries", "data", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                return [item for item in nested.values() if isinstance(item, dict)]

        if any(isinstance(item, dict) for item in value.values()):
            return [item for item in value.values() if isinstance(item, dict)]
        return [value]

    def _source_id(
        self, description: str, project: str, start_at: datetime | None, duration: str
    ) -> str:
        start_str = start_at.isoformat() if start_at else ""
        identifier = f"{project}|{description}|{start_str}|{duration}"
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
        return f"toggl:{digest[:24]}"

    def _format_title(self, description: str, project: str) -> str:
        if description and project:
            return f"{description} ({project})"
        if description:
            return description
        if project:
            return f"Time entry in {project}"
        return "Toggl time entry"

    def _content(
        self, description: str, project: str, duration: str, tags: list[str], billable: str
    ) -> str:
        parts = []
        if description:
            parts.append(f"Task: {description}")
        if project:
            parts.append(f"Project: {project}")
        if duration:
            parts.append(f"Duration: {duration}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if billable:
            billable_status = "Yes" if billable.lower() in ("yes", "true", "1") else "No"
            parts.append(f"Billable: {billable_status}")
        return "\n".join(parts)

    def _tags(self, item: dict[str, Any]) -> list[str]:
        # Toggl exports tags in a "Tags" column, comma-separated
        tags_str = self._first(item, "Tags", "tags", "tag")
        if not tags_str:
            return []

        tags: list[str] = []
        for tag in re.split(r",", tags_str):
            normalized = tag.strip().lower().replace(" ", "-")
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _parse_datetime_with_time(self, value: str) -> datetime | None:
        """Parse datetime string that combines date and time (e.g., '2025-01-15 14:30:00')."""
        if not value:
            return None
        try:
            # Try common datetime formats
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M:%S",
                "%Y/%m/%d %H:%M",
                "%d/%m/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M",
            ):
                try:
                    parsed = datetime.strptime(value, fmt)
                    return parsed.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            # Try ISO format
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return None

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        # Handle Unix timestamp
        if re.fullmatch(r"\d+(?:\.0+)?", value):
            try:
                return datetime.fromtimestamp(int(float(value)), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        # Handle ISO format and common date formats
        try:
            # Try ISO format first
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            # Try common date formats
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    parsed = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
