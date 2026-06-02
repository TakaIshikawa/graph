"""Adapter for GitHub starred repository JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GitHubStarredReposJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_starred_repos_json"

    @property
    def entity_types(self) -> list[str]:
        return ["repository"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "repository" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                records = _records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit(record, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        return result

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        repo = record.get("repo") if isinstance(record.get("repo"), dict) else record
        full_name = first(repo, "full_name", "fullName")
        name = first(repo, "name")
        owner = _owner(repo)
        if not full_name and owner and name:
            full_name = f"{owner}/{name}"
        url = first(repo, "html_url", "htmlUrl", "url")
        if not full_name and not url:
            return None
        description = first(repo, "description")
        snippet = first(repo, "readme", "readme_snippet", "readmeSnippet")
        language = _language(repo)
        topics = _topics(repo)
        stars = first(repo, "stargazers_count", "stargazersCount", "stars")
        starred_text = first(record, "starred_at", "starredAt") or first(repo, "starred_at", "starredAt")
        starred_at = parse_datetime(starred_text)
        now = datetime.now(timezone.utc)
        title = full_name or url or "GitHub repository"
        metadata = clean_metadata(
            {
                "id": first(repo, "id"),
                "full_name": full_name,
                "html_url": url,
                "url": url,
                "description": description,
                "language": language,
                "topics": topics,
                "stargazers_count": stars,
                "starred_at": starred_text,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="github_starred_repos_json",
            source_id=digest_source_id("github_starred_repos_json", first(repo, "id") or full_name or url or index),
            source_entity_type="repository",
            title=title,
            content=_content(title, description, snippet, url, language, topics),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=topics,
            created_at=starred_at or now,
            updated_at=starred_at or now,
        )


def _records(path: Path) -> list[dict[str, Any]]:
    parsed = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        for key in ("stars", "repositories", "items"):
            if isinstance(parsed.get(key), list):
                return [item for item in parsed[key] if isinstance(item, dict)]
        return [parsed]
    return []


def _owner(repo: dict[str, Any]) -> str:
    owner = repo.get("owner")
    if isinstance(owner, dict):
        return first(owner, "login", "name")
    return first(repo, "owner", "owner_login", "ownerLogin")


def _language(repo: dict[str, Any]) -> str:
    language = repo.get("language")
    if isinstance(language, dict):
        return first(language, "name")
    return first(repo, "language", "primaryLanguage")


def _topics(repo: dict[str, Any]) -> list[str]:
    raw = repo.get("topics") or repo.get("repositoryTopics")
    if isinstance(raw, dict):
        raw = raw.get("nodes", [])
    if isinstance(raw, list):
        values: list[str] = []
        for item in raw:
            topic = first(item.get("topic", item) if isinstance(item, dict) else {"name": item}, "name")
            if topic and topic not in values:
                values.append(topic)
        return values
    return split_values(raw)


def _content(title: str, description: str, snippet: str, url: str, language: str, topics: list[str]) -> str:
    parts = [title, description, snippet if snippet != description else ""]
    for label, value in (("Language", language), ("Topics", ", ".join(topics)), ("URL", url)):
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(part for part in parts if part)
