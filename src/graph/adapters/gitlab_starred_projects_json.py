"""Adapter for GitLab starred project JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GitlabStarredProjectsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "gitlab_starred_projects_json"

    @property
    def entity_types(self) -> list[str]:
        return ["repository"]

    def __init__(self, path: str | Path = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "repository" not in set(entity_types or self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit_from_record(record, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if root.is_dir():
            return sorted(child for child in root.rglob("*.json") if child.is_file())
        return []

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if not isinstance(parsed, dict):
            return []
        for key in ("projects", "starred_projects", "starredProjects", "items"):
            items = parsed.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return [parsed] if self._has_identity(parsed) else []

    def _unit_from_record(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        project_id = self._first(record, "id", "project_id", "projectId")
        path_with_namespace = self._first(record, "path_with_namespace", "pathWithNamespace", "full_path", "fullPath")
        name = self._first(record, "name", "name_with_namespace", "nameWithNamespace")
        web_url = self._first(record, "web_url", "webUrl", "url", "http_url_to_repo", "ssh_url_to_repo")
        if not project_id and not path_with_namespace and not name and not web_url:
            return None

        description = self._first(record, "description")
        namespace_path = self._namespace_path(record)
        topics = self._topics(record)
        star_count = self._parse_int(self._first(record, "star_count", "starCount", "star_count"))
        forks_count = self._parse_int(self._first(record, "forks_count", "forksCount", "forks_count"))
        default_branch = self._first(record, "default_branch", "defaultBranch")
        created_at_text = self._first(record, "created_at", "createdAt")
        last_activity_text = self._first(record, "last_activity_at", "lastActivityAt")
        created_at = self._parse_datetime(created_at_text)
        last_activity_at = self._parse_datetime(last_activity_text)
        now = datetime.now(timezone.utc)

        metadata = {
            "project_id": project_id,
            "path_with_namespace": path_with_namespace,
            "namespace_path": namespace_path,
            "name": name,
            "description": description,
            "web_url": web_url,
            "star_count": star_count,
            "forks_count": forks_count,
            "topics": topics,
            "default_branch": default_branch,
            "created_at": created_at.isoformat() if created_at else created_at_text,
            "last_activity_at": last_activity_at.isoformat() if last_activity_at else last_activity_text,
            "source_file": source_file,
            "record_index": index,
        }
        title = path_with_namespace or name or web_url or f"GitLab project {project_id}"
        return KnowledgeUnit(
            source_project="gitlab_starred_projects_json",
            source_id=self._source_id(project_id, path_with_namespace, web_url, index),
            source_entity_type="repository",
            title=title,
            content=self._content(title, description, web_url, namespace_path, topics, default_branch, last_activity_at),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=self._dedupe(["gitlab", *topics]),
            created_at=created_at or last_activity_at or now,
            updated_at=last_activity_at or created_at or now,
        )

    def _content(
        self,
        title: str,
        description: str,
        web_url: str,
        namespace_path: str,
        topics: list[str],
        default_branch: str,
        last_activity_at: datetime | None,
    ) -> str:
        parts = [title]
        if description:
            parts.append(description)
        if namespace_path:
            parts.append(f"Namespace: {namespace_path}")
        if topics:
            parts.append(f"Topics: {', '.join(topics)}")
        if default_branch:
            parts.append(f"Default branch: {default_branch}")
        if last_activity_at:
            parts.append(f"Last activity: {last_activity_at.isoformat()}")
        if web_url:
            parts.append(f"URL: {web_url}")
        return "\n".join(parts)

    def _namespace_path(self, record: dict[str, Any]) -> str:
        namespace = record.get("namespace")
        if isinstance(namespace, dict):
            return self._first(namespace, "full_path", "fullPath", "path", "name")
        return self._first(record, "namespace", "namespace_path", "namespacePath")

    def _topics(self, record: dict[str, Any]) -> list[str]:
        raw = record.get("topics")
        if raw is None:
            raw = record.get("tag_list") or record.get("tagList")
        if isinstance(raw, list):
            return self._dedupe(self._text(item) for item in raw)
        return self._dedupe(part.strip() for part in self._text(raw).replace(";", ",").replace("|", ",").split(",") if part.strip())

    def _has_identity(self, record: dict[str, Any]) -> bool:
        return bool(self._first(record, "id", "path_with_namespace", "pathWithNamespace", "name", "web_url", "url"))

    def _source_id(self, project_id: str, path_with_namespace: str, web_url: str, index: int) -> str:
        raw = project_id or path_with_namespace or web_url or str(index)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"gitlab_starred_projects_json:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).casefold(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.casefold())
            text = self._text(value)
            if text:
                return text
        return ""

    def _parse_int(self, value: str) -> int | None:
        if not value:
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    def _parse_datetime(self, value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _text(self, value: Any) -> str:
        if value is None or isinstance(value, (dict, list)):
            return ""
        return str(value).strip()

    def _dedupe(self, values: Any) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))
