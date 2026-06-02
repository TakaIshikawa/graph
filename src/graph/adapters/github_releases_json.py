"""Adapter for GitHub release JSON exports and API responses."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime, parse_int
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GithubReleasesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_releases_json"

    @property
    def entity_types(self) -> list[str]:
        return ["release"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "release" not in set(entity_types or self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for record in records:
                unit = self._unit_from_record(record, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("releases", "items", "nodes"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        release_id = parse_int(record.get("id") or record.get("databaseId"))
        tag_name = self._text(record.get("tag_name") or record.get("tagName"))
        name = self._text(record.get("name"))
        body = self._text(record.get("body") or record.get("description"))
        url = self._text(record.get("html_url") or record.get("url"))
        repository = self._repository(record)
        if release_id is None and not tag_name and not name and not url:
            return None
        created_at = parse_datetime(record.get("created_at") or record.get("createdAt"))
        published_at = parse_datetime(record.get("published_at") or record.get("publishedAt"))
        author = self._person(record.get("author") or record.get("user"))
        assets = self._assets(record.get("assets"))
        metadata = clean_metadata(
            {
                "id": release_id,
                "tag_name": tag_name,
                "name": name,
                "body": body,
                "draft": bool(record.get("draft")),
                "prerelease": bool(record.get("prerelease") or record.get("isPrerelease")),
                "author": author,
                "url": url,
                "assets": assets,
                "asset_names": [asset["name"] for asset in assets if asset.get("name")],
                "repository": repository,
                "created_at": created_at.isoformat() if created_at else self._text(record.get("created_at") or record.get("createdAt")),
                "published_at": published_at.isoformat() if published_at else self._text(record.get("published_at") or record.get("publishedAt")),
                "source_file": source_file,
                "record": record,
            }
        )
        now = datetime.now(timezone.utc)
        timestamp = published_at or created_at or now
        title = name or tag_name or (f"GitHub release {release_id}" if release_id is not None else url)
        return KnowledgeUnit(
            source_project=SourceProject.GITHUB_RELEASES_JSON,
            source_id=self._source_id(repository, release_id, tag_name, url, title),
            source_entity_type="release",
            title=title,
            content=self._content(title, tag_name, repository, url, body, assets),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(["github", "release", repository, tag_name])),
            created_at=created_at or timestamp,
            updated_at=timestamp,
        )

    def _source_id(self, repository: str, release_id: int | None, tag_name: str, url: str, title: str) -> str:
        if repository and tag_name:
            return f"github_releases_json:{repository}@{tag_name}"
        if release_id is not None:
            return f"github_releases_json:{release_id}"
        return digest_source_id("github_releases_json", repository, tag_name, url, title)

    def _content(self, title: str, tag_name: str, repository: str, url: str, body: str, assets: list[dict[str, Any]]) -> str:
        asset_names = [asset["name"] for asset in assets if asset.get("name")]
        parts = [
            title,
            f"Tag: {tag_name}" if tag_name else "",
            f"Repository: {repository}" if repository else "",
            f"URL: {url}" if url else "",
            f"Assets: {', '.join(asset_names)}" if asset_names else "",
            body,
        ]
        return "\n".join(part for part in parts if part)

    def _repository(self, record: dict[str, Any]) -> str:
        repository = record.get("repository")
        if isinstance(repository, dict):
            return self._text(repository.get("full_name") or repository.get("nameWithOwner") or repository.get("path_with_namespace") or repository.get("name"))
        return self._text(record.get("repository_full_name") or record.get("repositoryFullName") or repository)

    def _assets(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            value = value.get("nodes") or value.get("items")
        if not isinstance(value, list):
            return []
        assets: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            name = self._text(item.get("name"))
            assets.append(clean_metadata({"name": name, "url": self._text(item.get("browser_download_url") or item.get("downloadUrl") or item.get("url")), "size": parse_int(item.get("size"))}))
        return assets

    def _person(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("login") or value.get("username") or value.get("name"))
        return self._text(value)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
