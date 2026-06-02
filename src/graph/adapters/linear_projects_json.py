"""Adapter for Linear project JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime, parse_float, parse_int
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class LinearProjectsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "linear_projects_json"

    @property
    def entity_types(self) -> list[str]:
        return ["project"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "project" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._records(json.loads(path.read_text(encoding="utf-8-sig")))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records, start=1):
                record["_source_file"] = path.name
                record["_source_row"] = index
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
            for key in ("projects", "nodes", "data"):
                records = self._records(value.get(key))
                if records:
                    return records
            return [value]
        return []

    def _unit(self, record: dict[str, Any]) -> KnowledgeUnit | None:
        project_id = self._text(self._get(record, "id", "project_id", "projectId"))
        name = self._text(self._get(record, "name", "title"))
        team = self._person(record.get("team")) or self._text(self._get(record, "team", "teamName", "team_key"))
        state = self._text(self._get(record, "state", "status"))
        lead = self._person(self._get(record, "lead", "owner"))
        description = self._text(self._get(record, "description"))
        if not any([project_id, name, team, state]):
            return None
        created = parse_datetime(self._get(record, "createdAt", "created_at", "startDate", "start_date"))
        updated = parse_datetime(self._get(record, "updatedAt", "updated_at", "completedAt", "completed_at", "targetDate", "target_date"))
        url = self._text(self._get(record, "url", "appUrl", "app_url"))
        metadata = clean_metadata({"project_id": project_id, "name": name, "team": team, "state": state, "lead": lead, "description": description, "url": url, "source_url": url, "external_url": url, "target_date": self._date(record, "targetDate", "target_date"), "start_date": self._date(record, "startDate", "start_date"), "completed_date": self._date(record, "completedAt", "completed_at", "completedDate"), "progress": parse_float(self._get(record, "progress")), "issue_count": parse_int(self._get(record, "issueCount", "issue_count")), "source_file": record.get("_source_file"), "source_row": record.get("_source_row")})
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{project_id}" if project_id else digest_source_id(self.name, name, team), source_entity_type="project", title=name or "Linear project", content=self._content(name, description, metadata), content_type=ContentType.METADATA, metadata=metadata, tags=list(dict.fromkeys(tag for tag in ["linear", "project", team, state] if tag)), created_at=created or now, updated_at=updated or created or now)

    def _content(self, name: str, description: str, metadata: dict[str, Any]) -> str:
        parts = [name or "Linear project"]
        for key, label in (("team", "Team"), ("state", "State"), ("lead", "Lead"), ("progress", "Progress"), ("issue_count", "Issues"), ("url", "URL")):
            if key in metadata:
                parts.append(f"{label}: {metadata[key]}")
        if description:
            parts.append(description)
        return "\n".join(parts)

    def _date(self, record: dict[str, Any], *keys: str) -> str:
        parsed = parse_datetime(self._get(record, *keys))
        return parsed.date().isoformat() if parsed else self._text(self._get(record, *keys))

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
            return self._text(value.get("name") or value.get("key") or value.get("email") or value.get("id"))
        return ""

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
