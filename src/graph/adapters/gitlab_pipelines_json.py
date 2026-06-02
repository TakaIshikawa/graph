"""Adapter for GitLab pipeline JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime, parse_float, parse_int
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GitlabPipelinesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "gitlab_pipelines_json"

    @property
    def entity_types(self) -> list[str]:
        return ["pipeline"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "pipeline" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._records(json.loads(path.read_text(encoding="utf-8-sig")))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for record in records:
                record["_source_file"] = path.name
                unit = self._unit(record)
                if unit is None or (sync_at and unit.updated_at <= sync_at):
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("pipelines", "data", "nodes"):
                records = self._records(value.get(key))
                if records:
                    return records
            return [value]
        return []

    def _unit(self, record: dict[str, Any]) -> KnowledgeUnit | None:
        pipeline_id = self._text(self._get(record, "id", "pipeline_id"))
        iid = self._text(self._get(record, "iid"))
        project = self._project(record)
        ref = self._text(self._get(record, "ref", "branch"))
        sha = self._text(self._get(record, "sha", "commit_sha"))
        status = self._text(self._get(record, "status"))
        if not any([pipeline_id, iid, project, ref, sha, status]):
            return None
        created = parse_datetime(self._get(record, "created_at", "createdAt"))
        updated = parse_datetime(self._get(record, "updated_at", "updatedAt", "finished_at", "finishedAt")) or created
        url = self._text(self._get(record, "web_url", "webUrl", "url"))
        user = self._person(record.get("user"))
        metadata = clean_metadata({"pipeline_id": pipeline_id, "iid": iid, "project_path": project, "ref": ref, "sha": sha, "status": status, "source": self._text(self._get(record, "source")), "web_url": url, "source_url": url, "external_url": url, "duration": parse_float(self._get(record, "duration")), "queued_duration": parse_float(self._get(record, "queued_duration", "queuedDuration")), "created_at": created.isoformat() if created else self._text(self._get(record, "created_at", "createdAt")), "updated_at": updated.isoformat() if updated else self._text(self._get(record, "updated_at", "updatedAt")), "finished_at": self._date(record, "finished_at", "finishedAt"), "user": user, "source_file": record.get("_source_file")})
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{project}:{pipeline_id}" if pipeline_id and project else digest_source_id(self.name, project, sha, ref, created), source_entity_type="pipeline", title=f"GitLab pipeline {project} {ref}".strip(), content=self._content(metadata), content_type=ContentType.METADATA, metadata=metadata, tags=list(dict.fromkeys(tag for tag in ["gitlab", "pipeline", status, ref] if tag)), created_at=created or now, updated_at=updated or created or now)

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [f"GitLab pipeline {metadata.get('project_path', '')}".strip()]
        for key, label in (("ref", "Ref"), ("status", "Status"), ("sha", "SHA"), ("duration", "Duration"), ("web_url", "URL")):
            if key in metadata:
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)

    def _project(self, record: dict[str, Any]) -> str:
        project = record.get("project")
        if isinstance(project, dict):
            return self._text(project.get("path_with_namespace") or project.get("full_path") or project.get("path"))
        return self._text(self._get(record, "project_path", "projectPath", "project"))

    def _date(self, record: dict[str, Any], *keys: str) -> str:
        parsed = parse_datetime(self._get(record, *keys))
        return parsed.isoformat() if parsed else self._text(self._get(record, *keys))

    def _get(self, record: dict[str, Any], *keys: str) -> Any:
        compact = {"".join(ch for ch in str(k).casefold() if ch.isalnum()): v for k, v in record.items()}
        for key in keys:
            if key in record:
                return record[key]
            value = compact.get("".join(ch for ch in key.casefold() if ch.isalnum()))
            if value is not None:
                return value
        return None

    def _person(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("username") or value.get("name"))
        return self._text(value)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
