"""Adapter for GitHub stars CSV exports."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class GithubStarsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_stars_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["repository", "owner", "topic"]

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
        repositories: list[KnowledgeUnit] = []

        for path in self._iter_paths(".csv"):
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                repositories.append(unit)

        owners = self._owner_units(repositories) if "owner" in allowed_types else []
        topics = self._topic_units(repositories) if "topic" in allowed_types else []
        if "repository" in allowed_types:
            result.units.extend(repositories)
        if "owner" in allowed_types:
            result.units.extend(owners)
        if "topic" in allowed_types:
            result.units.extend(topics)
        if "owner" in allowed_types and "repository" in allowed_types:
            result.edges.extend(self._owner_repository_edges(owners))
        if "topic" in allowed_types and "repository" in allowed_types:
            result.edges.extend(self._topic_repository_edges(topics))
        entity_order = {entity_type: index for index, entity_type in enumerate(self.entity_types)}
        result.units.sort(key=lambda unit: (entity_order.get(unit.source_entity_type, 99), unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        full_name = self._first(row, "full_name", "repo", "repository", "repository name", "name")
        html_url = self._first(row, "html_url", "url")
        if not full_name and html_url:
            full_name = self._full_name_from_url(html_url)
        if not full_name and not html_url:
            return None
        title = full_name or html_url
        description = self._first(row, "description") or html_url
        topics = self._parse_topics(self._first(row, "topics", "topic"))
        starred_at_text = self._first(row, "starred_at", "created_at")
        starred_at = self._parse_datetime(starred_at_text)
        now = datetime.now(timezone.utc)
        owner, repo = self._repo_parts(full_name, self._first(row, "owner"))
        stars = self._parse_int(self._first(row, "stargazers_count", "stars", "star_count"))
        metadata = {
            "full_name": full_name,
            "owner": owner,
            "repo": repo,
            "description": self._first(row, "description"),
            "url": html_url,
            "source_url": html_url,
            "external_url": html_url,
            "language": self._first(row, "language"),
            "topics": topics,
            "stars": stars,
            "starred_at": starred_at_text,
            "stargazers_count": stars,
            "archived": self._parse_bool(self._first(row, "archived", "is_archived")),
            "private": self._parse_bool(self._first(row, "private", "is_private")),
            "source_file": source_file,
        }
        return KnowledgeUnit(
            source_project=SourceProject.GITHUB_STARS_CSV,
            source_id=self._repository_source_id(full_name, html_url),
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

    def _repository_source_id(self, full_name: str, url: str) -> str:
        raw = full_name.casefold() if full_name else url.strip().casefold()
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"github_stars_csv:repository:{digest}"

    def _full_name_from_url(self, url: str) -> str:
        text = url.strip().rstrip("/")
        marker = "github.com/"
        if marker not in text.casefold():
            return ""
        path = text[text.casefold().index(marker) + len(marker):]
        parts = [part for part in path.split("/") if part]
        if len(parts) < 2:
            return ""
        return f"{parts[0]}/{parts[1]}"

    def _repo_parts(self, full_name: str, owner: str) -> tuple[str, str]:
        if "/" in full_name:
            parsed_owner, repo = full_name.split("/", 1)
            return owner or parsed_owner, repo
        return owner, full_name

    def _owner_units(self, repositories: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for repository in repositories:
            owner = str(repository.metadata.get("owner") or "").strip()
            if not owner:
                continue
            normalized = self._normalize_owner(owner)
            grouped.setdefault(normalized, []).append(repository)
            names.setdefault(normalized, owner)

        units: list[KnowledgeUnit] = []
        for normalized, owner_repositories in sorted(grouped.items()):
            languages = sorted(
                {str(repository.metadata.get("language") or "") for repository in owner_repositories if repository.metadata.get("language")}
            )
            topics = sorted(
                {
                    topic
                    for repository in owner_repositories
                    for topic in repository.metadata.get("topics", [])
                    if topic
                }
            )
            stargazer_counts = [
                repository.metadata.get("stargazers_count")
                for repository in owner_repositories
                if isinstance(repository.metadata.get("stargazers_count"), int)
            ]
            starred_dates = [
                parsed
                for repository in owner_repositories
                if (parsed := self._parse_datetime(str(repository.metadata.get("starred_at") or ""))) is not None
            ]
            first_starred = min(starred_dates).isoformat() if starred_dates else None
            latest_starred = max(starred_dates).isoformat() if starred_dates else None
            repository_source_ids = sorted(repository.source_id for repository in owner_repositories)
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GITHUB_STARS_CSV,
                    source_id=self._owner_source_id(normalized),
                    source_entity_type="owner",
                    title=names[normalized],
                    content=f"GitHub starred repository owner: {names[normalized]}\nRepositories: {len(owner_repositories)}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "owner": names[normalized],
                        "normalized_owner": normalized,
                        "repo_count": len(owner_repositories),
                        "languages": languages,
                        "topics": topics,
                        "stargazers_count": sum(stargazer_counts) if stargazer_counts else None,
                        "first_starred_at": first_starred,
                        "latest_starred_at": latest_starred,
                        "repository_source_ids": repository_source_ids,
                    },
                    tags=["github", "owner", names[normalized]],
                    created_at=min(repository.created_at for repository in owner_repositories),
                    updated_at=max(repository.updated_at for repository in owner_repositories),
                )
            )
        return units

    def _owner_repository_edges(self, owners: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for owner in owners:
            for repository_source_id in owner.metadata.get("repository_source_ids") or []:
                edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(owner.source_id, str(repository_source_id)),
                        from_unit_id=owner.source_id,
                        to_unit_id=str(repository_source_id),
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.GITHUB_STARS_CSV.value,
                            "relation_type": "owner_contains_repository",
                        },
                    )
                )
        return edges

    def _topic_units(self, repositories: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for repository in repositories:
            for topic in repository.metadata.get("topics", []) or []:
                topic_text = str(topic).strip()
                if not topic_text:
                    continue
                normalized = self._normalize_topic(topic_text)
                grouped.setdefault(normalized, []).append(repository)
                names.setdefault(normalized, topic_text)

        units: list[KnowledgeUnit] = []
        for normalized, topic_repositories in sorted(grouped.items()):
            repository_source_ids = sorted({repository.source_id for repository in topic_repositories})
            owners = sorted(
                {
                    str(repository.metadata.get("owner") or "")
                    for repository in topic_repositories
                    if repository.metadata.get("owner")
                },
                key=lambda value: value.casefold(),
            )
            languages = sorted(
                {
                    str(repository.metadata.get("language") or "")
                    for repository in topic_repositories
                    if repository.metadata.get("language")
                }
            )
            starred_dates = [
                parsed
                for repository in topic_repositories
                if (parsed := self._parse_datetime(str(repository.metadata.get("starred_at") or ""))) is not None
            ]
            first_starred = min(starred_dates).isoformat() if starred_dates else None
            latest_starred = max(starred_dates).isoformat() if starred_dates else None
            topic_name = names[normalized]
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GITHUB_STARS_CSV,
                    source_id=self._topic_source_id(normalized),
                    source_entity_type="topic",
                    title=topic_name,
                    content=f"GitHub starred repository topic: {topic_name}\nRepositories: {len(repository_source_ids)}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "topic": topic_name,
                        "normalized_topic": normalized,
                        "repo_count": len(repository_source_ids),
                        "repository_source_ids": repository_source_ids,
                        "owners": owners,
                        "languages": languages,
                        "first_starred_at": first_starred,
                        "latest_starred_at": latest_starred,
                    },
                    tags=["github", "topic", topic_name],
                    created_at=min(repository.created_at for repository in topic_repositories),
                    updated_at=max(repository.updated_at for repository in topic_repositories),
                )
            )
        return units

    def _topic_repository_edges(self, topics: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for topic in topics:
            for repository_source_id in topic.metadata.get("repository_source_ids") or []:
                edges.append(
                    KnowledgeEdge(
                        id=self._topic_edge_id(topic.source_id, str(repository_source_id)),
                        from_unit_id=topic.source_id,
                        to_unit_id=str(repository_source_id),
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.GITHUB_STARS_CSV.value,
                            "relation_type": "topic_contains_repository",
                        },
                    )
                )
        return edges

    def _normalize_owner(self, owner: str) -> str:
        return owner.strip().casefold()

    def _normalize_topic(self, topic: str) -> str:
        return topic.strip().casefold()

    def _owner_source_id(self, normalized_owner: str) -> str:
        digest = hashlib.sha256(normalized_owner.encode("utf-8")).hexdigest()[:24]
        return f"github_stars_csv:owner:{digest}"

    def _topic_source_id(self, normalized_topic: str) -> str:
        digest = hashlib.sha256(normalized_topic.encode("utf-8")).hexdigest()[:24]
        return f"github_stars_csv:topic:{digest}"

    def _edge_id(self, owner_source_id: str, repository_source_id: str) -> str:
        digest = hashlib.sha256("|".join((owner_source_id, repository_source_id, "owner_contains_repository")).encode("utf-8")).hexdigest()[:24]
        return f"github-stars-csv-owner-contains-{digest}"

    def _topic_edge_id(self, topic_source_id: str, repository_source_id: str) -> str:
        digest = hashlib.sha256("|".join((topic_source_id, repository_source_id, "topic_contains_repository")).encode("utf-8")).hexdigest()[:24]
        return f"github-stars-csv-topic-contains-{digest}"

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
        lowered = {str(key).casefold(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.casefold())
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_int(self, value: str) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_bool(self, value: str) -> bool | None:
        text = str(value or "").strip().casefold()
        if not text:
            return None
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
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
