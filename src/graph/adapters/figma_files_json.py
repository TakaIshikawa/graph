"""Adapter for Figma file JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime, parse_int
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class FigmaFilesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "figma_files_json"

    @property
    def entity_types(self) -> list[str]:
        return ["file"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "file" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                records = _records(json.loads(path.read_text(encoding="utf-8-sig")))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit(record, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        key = _text(record.get("key") or record.get("file_key") or record.get("id"))
        name = _text(record.get("name") or record.get("title"))
        url = _text(record.get("url") or record.get("file_url"))
        if not any((key, name, url)):
            return None
        project = _name(record.get("project")) or _text(record.get("project_name"))
        team = _name(record.get("team")) or _text(record.get("team_name"))
        description = _text(record.get("description"))
        last_modified = parse_datetime(record.get("last_modified") or record.get("lastModified") or record.get("updated_at"))
        metadata = clean_metadata(
            {
                "key": key,
                "name": name,
                "url": url,
                "project": project,
                "team": team,
                "thumbnail_url": _text(record.get("thumbnail_url") or record.get("thumbnailUrl")),
                "last_modified": last_modified.isoformat() if last_modified else _text(record.get("last_modified") or record.get("lastModified")),
                "version_count": parse_int(record.get("version_count") or record.get("versions")),
                "description": description,
                "source_file": source_file,
            }
        )
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.FIGMA_FILES_JSON,
            source_id=f"figma_files_json:{key}" if key else digest_source_id("figma_files_json", name, url, source_file, index),
            source_entity_type="file",
            title=name or key or "Figma file",
            content=_content(name, description, metadata),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=[tag for tag in dict.fromkeys(["figma", project, team]) if tag],
            created_at=last_modified or now,
            updated_at=last_modified or now,
        )


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("files", "items", "data", "results"):
            items = value.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return [value]
    return []


def _name(value: Any) -> str:
    if isinstance(value, dict):
        return _text(value.get("name") or value.get("id"))
    return _text(value)


def _content(name: str, description: str, metadata: dict[str, Any]) -> str:
    parts = [name, description]
    for key, label in (("project", "Project"), ("team", "Team"), ("url", "URL")):
        if metadata.get(key):
            parts.append(f"{label}: {metadata[key]}")
    return "\n".join(part for part in parts if part)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()
