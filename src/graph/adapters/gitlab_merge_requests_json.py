"""Adapter for GitLab merge request JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GitlabMergeRequestsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "gitlab_merge_requests_json"

    @property
    def entity_types(self) -> list[str]:
        return ["merge_request"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "merge_request" not in set(entity_types or self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
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

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.json") if child.is_file())

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("merge_requests", "mrs", "items", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        iid = self._text(record.get("iid") or record.get("id"))
        project = self._text(record.get("path_with_namespace") or record.get("project_path") or record.get("project_id"))
        title = self._text(record.get("title"))
        description = self._text(record.get("description"))
        state = self._text(record.get("state"))
        web_url = self._text(record.get("web_url") or record.get("url"))
        author = self._person(record.get("author"))
        assignees = self._people(record.get("assignees") or record.get("assignee"))
        labels = self._list(record.get("labels"))
        source_branch = self._text(record.get("source_branch"))
        target_branch = self._text(record.get("target_branch"))
        merged_at = self._parse_datetime(record.get("merged_at"))
        created_at = self._parse_datetime(record.get("created_at"))
        updated_at = self._parse_datetime(record.get("updated_at")) or merged_at or created_at
        if not title and not web_url:
            return None
        metadata = {
            "iid": iid,
            "project": project,
            "title": title,
            "description": description,
            "state": state,
            "web_url": web_url,
            "author": author,
            "assignees": assignees,
            "labels": labels,
            "source_branch": source_branch,
            "target_branch": target_branch,
            "merged_at": merged_at.isoformat() if merged_at else self._text(record.get("merged_at")),
            "created_at": created_at.isoformat() if created_at else self._text(record.get("created_at")),
            "updated_at": updated_at.isoformat() if updated_at else self._text(record.get("updated_at")),
            "source_file": source_file,
            "record": record,
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.GITLAB_MERGE_REQUESTS_JSON,
            source_id=self._source_id(project, iid, web_url, title),
            source_entity_type="merge_request",
            title=title or web_url,
            content=self._content(project, title, state, source_branch, target_branch, web_url),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(["gitlab", "merge_request", *labels])),
            created_at=created_at or now,
            updated_at=updated_at or now,
        )

    def _content(self, project: str, title: str, state: str, source_branch: str, target_branch: str, url: str) -> str:
        branch = f"{source_branch} -> {target_branch}" if source_branch or target_branch else ""
        parts = [title, f"Project: {project}" if project else "", f"State: {state}" if state else "", f"Branches: {branch}" if branch else "", f"URL: {url}" if url else ""]
        return "\n".join(part for part in parts if part)

    def _source_id(self, project: str, iid: str, url: str, title: str) -> str:
        digest = hashlib.sha256((url or f"{project}|{iid}" or title).encode("utf-8")).hexdigest()[:24]
        return f"gitlab_merge_requests_json:{digest}"

    def _people(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [self._person(item) for item in value if self._person(item)]
        person = self._person(value)
        return [person] if person else []

    def _person(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("username") or value.get("name"))
        return self._text(value)

    def _list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [self._text(item) for item in value if self._text(item)]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return []

    def _parse_datetime(self, value: Any) -> datetime | None:
        text = self._text(value)
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
        return "" if value is None else str(value).strip()
