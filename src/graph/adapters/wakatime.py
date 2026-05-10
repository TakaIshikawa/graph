"""Adapter for WakaTime coding activity tracking exports."""

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


class WakaTimeAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "wakatime"

    @property
    def entity_types(self) -> list[str]:
        return ["coding_session"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "coding_session" not in entity_types:
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
            language = self._first(item, "language", "Language")
            project = self._first(item, "project", "Project")
            if not language and not project:
                continue

            time_str = self._first(item, "time", "timestamp", "date", "Time", "Date")
            time_at = self._parse_datetime(time_str)
            if sync_at and time_at and time_at <= sync_at:
                continue

            file_path = self._first(item, "file", "File", "entity")
            branch = self._first(item, "branch", "Branch")
            editor = self._first(item, "editor", "Editor")
            time_spent = self._first(item, "duration", "time_spent", "seconds", "Duration")

            # Filter out system files
            if self._is_system_file(file_path):
                continue

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.WAKATIME,
                    source_id=self._source_id(project, language, time_str, file_path),
                    source_entity_type="coding_session",
                    title=self._format_title(project, language, time_str),
                    content=self._content(project, language, file_path, editor, time_spent),
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "language": language,
                        "project": project,
                        "file": file_path,
                        "branch": branch,
                        "editor": editor,
                        "time_spent_seconds": time_spent,
                        "time": time_str,
                    },
                    tags=self._tags(language, project),
                    created_at=time_at or datetime.now(timezone.utc),
                    updated_at=time_at or datetime.now(timezone.utc),
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

        for key in ("data", "days", "sessions", "activities", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                return [item for item in nested.values() if isinstance(item, dict)]

        if any(isinstance(item, dict) for item in value.values()):
            return [item for item in value.values() if isinstance(item, dict)]
        return [value]

    def _is_system_file(self, file_path: str) -> bool:
        """Filter out common system/config files that aren't meaningful coding activity."""
        if not file_path:
            return False
        lower_path = file_path.lower()
        # System directories and files to exclude
        excluded_patterns = [
            "node_modules/",
            ".git/",
            "__pycache__/",
            ".venv/",
            "venv/",
            ".idea/",
            ".vscode/",
            "dist/",
            "build/",
        ]
        return any(pattern in lower_path for pattern in excluded_patterns)

    def _source_id(self, project: str, language: str, time: str, file_path: str) -> str:
        identifier = f"{project}|{language}|{time}|{file_path}"
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
        return f"wakatime:{digest[:24]}"

    def _format_title(self, project: str, language: str, time: str) -> str:
        parts = []
        if language:
            parts.append(language)
        if project:
            parts.append(f"in {project}")
        if time:
            parts.append(f"on {time}")
        if parts:
            return " ".join(parts)
        return "WakaTime coding session"

    def _content(
        self, project: str, language: str, file_path: str, editor: str, time_spent: str
    ) -> str:
        parts = []
        if project:
            parts.append(f"Project: {project}")
        if language:
            parts.append(f"Language: {language}")
        if file_path:
            parts.append(f"File: {file_path}")
        if editor:
            parts.append(f"Editor: {editor}")
        if time_spent:
            parts.append(f"Time Spent: {self._format_duration(time_spent)}")
        return "\n".join(parts)

    def _format_duration(self, seconds_str: str) -> str:
        """Convert seconds to human-readable duration."""
        try:
            seconds = int(float(seconds_str))
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

    def _tags(self, language: str, project: str) -> list[str]:
        tags: list[str] = []
        if language:
            normalized_language = language.strip().lower().replace(" ", "-")
            if normalized_language:
                tags.append(normalized_language)
        if project:
            normalized_project = project.strip().lower().replace(" ", "-")
            if normalized_project and normalized_project not in tags:
                tags.append(normalized_project)
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
