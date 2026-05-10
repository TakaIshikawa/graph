"""Adapter for RescueTime productivity tracking exports."""

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


class RescueTimeAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "rescuetime"

    @property
    def entity_types(self) -> list[str]:
        return ["activity"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "activity" not in entity_types:
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
            activity = self._first(item, "Activity", "activity", "application", "app")
            if not activity:
                continue

            date_str = self._first(item, "Date", "date", "timestamp")
            date_time = self._parse_datetime(date_str)
            if sync_at and date_time and date_time <= sync_at:
                continue

            time_spent = self._first(item, "Time Spent (seconds)", "time_spent", "duration")
            category = self._first(item, "Category", "category")
            productivity = self._first(item, "Productivity", "productivity", "productivity_score")

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.RESCUETIME,
                    source_id=self._source_id(activity, date_str, time_spent),
                    source_entity_type="activity",
                    title=self._format_title(activity, date_str),
                    content=self._content(activity, category, time_spent, productivity),
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "activity": activity,
                        "category": category,
                        "time_spent_seconds": time_spent,
                        "productivity": productivity,
                        "date": date_str,
                    },
                    tags=self._tags(category, productivity),
                    created_at=date_time or datetime.now(timezone.utc),
                    updated_at=date_time or datetime.now(timezone.utc),
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

        for key in ("activities", "rows", "data", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                return [item for item in nested.values() if isinstance(item, dict)]

        if any(isinstance(item, dict) for item in value.values()):
            return [item for item in value.values() if isinstance(item, dict)]
        return [value]

    def _source_id(self, activity: str, date: str, time_spent: str) -> str:
        identifier = f"{activity}|{date}|{time_spent}"
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
        return f"rescuetime:{digest[:24]}"

    def _format_title(self, activity: str, date: str) -> str:
        if date:
            return f"{activity} on {date}"
        return activity or "RescueTime activity"

    def _content(self, activity: str, category: str, time_spent: str, productivity: str) -> str:
        parts = []
        if activity:
            parts.append(f"Activity: {activity}")
        if category:
            parts.append(f"Category: {category}")
        if time_spent:
            parts.append(f"Time Spent: {self._format_duration(time_spent)}")
        if productivity:
            parts.append(f"Productivity: {productivity}")
        return "\n".join(parts)

    def _format_duration(self, seconds_str: str) -> str:
        """Convert seconds to human-readable duration."""
        try:
            seconds = int(seconds_str)
            if seconds < 60:
                return f"{seconds}s"
            minutes = seconds // 60
            if minutes < 60:
                return f"{minutes}m"
            hours = minutes // 60
            remaining_minutes = minutes % 60
            if remaining_minutes > 0:
                return f"{hours}h {remaining_minutes}m"
            return f"{hours}h"
        except (ValueError, TypeError):
            return seconds_str

    def _tags(self, category: str, productivity: str) -> list[str]:
        tags: list[str] = []
        if category:
            normalized_category = category.strip().lower().replace(" ", "-")
            if normalized_category:
                tags.append(normalized_category)
        if productivity:
            productivity_tag = f"productivity-{productivity.lower().replace(' ', '-')}"
            if productivity_tag not in tags:
                tags.append(productivity_tag)
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
