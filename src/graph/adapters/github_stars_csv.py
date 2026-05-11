"""Adapter for GitHub stars CSV exports."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GithubStarsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_stars_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["repository"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "repository" not in entity_types:
            return result
        sync_at = self._sync_datetime(since) if since else None

        for path in self._iter_paths(".csv"):
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit_from_row(self, row: dict[str, Any]) -> KnowledgeUnit | None:
        full_name = self._first(row, "full_name", "repo", "repository")
        html_url = self._first(row, "html_url", "url")
        if not full_name and not html_url:
            return None
        title = full_name or html_url
        description = self._first(row, "description") or html_url
        topics = self._parse_topics(self._first(row, "topics", "topic"))
        starred_at_text = self._first(row, "starred_at", "created_at")
        starred_at = self._parse_datetime(starred_at_text)
        now = datetime.now(timezone.utc)
        metadata = {
            "full_name": full_name,
            "description": self._first(row, "description"),
            "source_url": html_url,
            "external_url": html_url,
            "language": self._first(row, "language"),
            "topics": topics,
            "owner": self._first(row, "owner") or full_name.split("/", 1)[0],
            "starred_at": starred_at_text,
            "stargazers_count": self._parse_int(self._first(row, "stargazers_count", "stars")),
        }
        return KnowledgeUnit(
            source_project=SourceProject.GITHUB_STARS_CSV,
            source_id=f"github_stars_csv:{full_name or html_url}",
            source_entity_type="repository",
            title=title,
            content=self._content(title, description, html_url, topics),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=topics,
            created_at=starred_at or now,
            updated_at=starred_at or now,
        )

    def _content(self, title: str, description: str, url: str, topics: list[str]) -> str:
        parts = [title]
        if description and description != title:
            parts.append(description)
        if url:
            parts.append(f"URL: {url}")
        if topics:
            parts.append(f"Topics: {', '.join(topics)}")
        return "\n".join(parts)

    def _parse_topics(self, value: str) -> list[str]:
        if not value:
            return []
        parsed: Any
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        raw = parsed if isinstance(parsed, list) else value.replace(";", ",").replace("|", ",").split(",")
        topics: list[str] = []
        for item in raw:
            topic = str(item).strip().strip("\"'").lower()
            if topic and topic not in topics:
                topics.append(topic)
        return topics

    def _iter_paths(self, suffix: str) -> list[Path]:
        path = Path(self.path).expanduser() if self.path else None
        if path is None:
            return []
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(child for child in path.rglob(f"*{suffix}") if child.is_file())
        return []

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            return [{str(k).strip(): v for k, v in row.items() if k is not None} for row in reader]

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_int(self, value: str) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
