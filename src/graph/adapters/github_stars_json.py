"""Adapter for GitHub starred repository JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GithubStarsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_stars_json"

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
        allowed = set(entity_types or self.entity_types)
        if not allowed.intersection(self.entity_types):
            return result
        sync_at = self._sync_datetime(since) if since else None

        for path in self._iter_paths():
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit_from_record(record, path, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _iter_paths(self) -> list[Path]:
        path = Path(self.path).expanduser() if self.path else None
        if path is None:
            return []
        if path.is_file() and path.suffix.lower() == ".json":
            return [path]
        if path.is_dir():
            return sorted(child for child in path.rglob("*.json") if child.is_file())
        return []

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if not isinstance(parsed, dict):
            return []
        for key in ("stars", "repositories"):
            items = parsed.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return [parsed] if self._first(parsed, "full_name", "name", "html_url", "url") else []

    def _unit_from_record(self, record: dict[str, Any], path: Path, index: int) -> KnowledgeUnit | None:
        repo_record = record.get("repo") if isinstance(record.get("repo"), dict) else record
        full_name = self._first(repo_record, "full_name", "fullName")
        owner = self._owner(repo_record)
        repo = self._repo_name(repo_record, full_name)
        if not full_name and owner and repo:
            full_name = f"{owner}/{repo}"
        html_url = self._first(repo_record, "html_url", "htmlUrl", "url")
        if not full_name and not html_url:
            return None

        description = self._first(repo_record, "description")
        language = self._language(repo_record)
        topics = self._topics(repo_record)
        starred_at_text = self._first(record, "starred_at", "starredAt", "created_at", "createdAt") or self._first(repo_record, "starred_at", "starredAt")
        starred_at = self._parse_datetime(starred_at_text)
        now = datetime.now(timezone.utc)
        title = full_name or html_url

        metadata = {
            "full_name": full_name,
            "owner": owner or full_name.split("/", 1)[0],
            "repo": repo or full_name.rsplit("/", 1)[-1],
            "html_url": html_url,
            "language": language,
            "topics": topics,
            "starred_at": starred_at_text,
            "description": description,
            "source_file": str(path),
            "record_index": index,
        }
        return KnowledgeUnit(
            source_project=SourceProject.GITHUB_STARS_CSV,
            source_id=self._source_id(full_name, html_url, index),
            source_entity_type="repository",
            title=title,
            content=self._content(title, owner, repo, description, html_url, language, topics),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=topics,
            created_at=starred_at or now,
            updated_at=starred_at or now,
        )

    def _content(self, title: str, owner: str, repo: str, description: str, url: str, language: str, topics: list[str]) -> str:
        parts = [title]
        if owner and repo and title != f"{owner}/{repo}":
            parts.append(f"Repository: {owner}/{repo}")
        if description:
            parts.append(description)
        if language:
            parts.append(f"Language: {language}")
        if topics:
            parts.append(f"Topics: {', '.join(topics)}")
        if url:
            parts.append(f"URL: {url}")
        return "\n".join(parts)

    def _owner(self, record: dict[str, Any]) -> str:
        owner = record.get("owner")
        if isinstance(owner, dict):
            return self._first(owner, "login", "name")
        return self._first(record, "owner", "owner_login", "ownerLogin")

    def _repo_name(self, record: dict[str, Any], full_name: str) -> str:
        name = self._first(record, "name", "repo", "repository")
        if name:
            return name
        return full_name.rsplit("/", 1)[-1] if "/" in full_name else ""

    def _language(self, record: dict[str, Any]) -> str:
        language = record.get("language")
        if isinstance(language, dict):
            return self._first(language, "name")
        return self._first(record, "language", "primaryLanguage")

    def _topics(self, record: dict[str, Any]) -> list[str]:
        raw = record.get("topics") or record.get("repositoryTopics")
        if isinstance(raw, dict):
            raw = raw.get("nodes")
        if isinstance(raw, list):
            topics = []
            for item in raw:
                if isinstance(item, dict):
                    topic = self._first(item.get("topic", {}) if isinstance(item.get("topic"), dict) else item, "name")
                else:
                    topic = str(item).strip()
                if topic and topic not in topics:
                    topics.append(topic)
            return topics
        text = str(raw or "").strip()
        return [item.strip() for item in text.replace(";", ",").replace("|", ",").split(",") if item.strip()]

    def _source_id(self, full_name: str, html_url: str, index: int) -> str:
        raw = full_name or html_url or str(index)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"github_stars_json:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).casefold(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.casefold())
            if value is not None and not isinstance(value, (dict, list)) and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
