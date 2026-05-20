"""Adapter for Linear cycle JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime, parse_float, parse_int
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class LinearCyclesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "linear_cycles_json"

    @property
    def entity_types(self) -> list[str]:
        return ["cycle"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "cycle" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        records: list[dict[str, Any]] = []
        for path in iter_paths(self.path, {".json"}):
            try:
                for record in self._read_records(path):
                    record["_source_file"] = path.name
                    records.append(record)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue

        for record in records:
            unit = self._unit(record)
            if unit is None:
                continue
            if sync_at and unit.updated_at <= sync_at:
                continue
            result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._cycles(parsed)

    def _cycles(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []

        for key in ("cycles", "data", "nodes"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                records = self._cycles(nested)
                if records:
                    return records

        return [value]

    def _unit(self, record: dict[str, Any]) -> KnowledgeUnit | None:
        cycle_id = self._text(self._get(record, "id", "cycle_id", "cycleId"))
        number = self._text(self._get(record, "number", "cycle_number", "cycleNumber"))
        name = self._text(self._get(record, "name", "title"))
        team = self._team(record.get("team") or record.get("teamData"))
        team_key = self._text(self._get(record, "teamKey", "team_key")) or team.get("key", "")
        team_name = self._text(self._get(record, "teamName", "team_name")) or team.get("name", "")
        description = self._text(self._get(record, "description"))
        if not any([cycle_id, number, name, team_key, team_name]):
            return None

        starts_at = parse_datetime(self._get(record, "startsAt", "starts_at", "startDate", "start_date"))
        ends_at = parse_datetime(self._get(record, "endsAt", "ends_at", "endDate", "end_date"))
        completed_at = parse_datetime(self._get(record, "completedAt", "completed_at"))
        created_at = parse_datetime(self._get(record, "createdAt", "created_at")) or starts_at
        updated_at = parse_datetime(self._get(record, "updatedAt", "updated_at")) or completed_at or ends_at or starts_at or created_at

        progress = parse_float(self._get(record, "progress"))
        archived = self._bool(self._get(record, "archived", "isArchived", "is_archived"))
        url = self._text(self._get(record, "url", "appUrl", "app_url"))

        metadata = clean_metadata(
            {
                "cycle_id": cycle_id,
                "number": parse_int(number) if number else None,
                "name": name,
                "team_key": team_key,
                "team_name": team_name,
                "starts_at": starts_at.isoformat() if starts_at else self._text(self._get(record, "startsAt", "starts_at")),
                "ends_at": ends_at.isoformat() if ends_at else self._text(self._get(record, "endsAt", "ends_at")),
                "completed_at": completed_at.isoformat() if completed_at else self._text(self._get(record, "completedAt", "completed_at")),
                "created_at": created_at.isoformat() if created_at else self._text(self._get(record, "createdAt", "created_at")),
                "updated_at": updated_at.isoformat() if updated_at else self._text(self._get(record, "updatedAt", "updated_at")),
                "progress": progress,
                "issue_count": parse_int(self._get(record, "issueCount", "issue_count", "totalIssueCount", "total_issue_count")),
                "completed_issue_count": parse_int(self._get(record, "completedIssueCount", "completed_issue_count")),
                "started_issue_count": parse_int(self._get(record, "startedIssueCount", "started_issue_count")),
                "uncompleted_issue_count": parse_int(self._get(record, "uncompletedIssueCount", "uncompleted_issue_count")),
                "scope": parse_float(self._get(record, "scope")),
                "completed_scope": parse_float(self._get(record, "completedScope", "completed_scope")),
                "started_scope": parse_float(self._get(record, "startedScope", "started_scope")),
                "description": description,
                "archived": archived,
                "url": url,
                "source_url": url,
                "external_url": url,
                "source_file": record.get("_source_file"),
            }
        )
        now = datetime.now(timezone.utc)
        title = self._title(name, number, team_key or team_name)
        return KnowledgeUnit(
            source_project="linear_cycles_json",
            source_id=self._source_id(cycle_id, team_key or team_name, number, name),
            source_entity_type="cycle",
            title=title,
            content=self._content(title, description, metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["linear", "cycle", team_key, team_name] if tag)),
            created_at=created_at or starts_at or now,
            updated_at=updated_at or now,
        )

    def _source_id(self, cycle_id: str, team: str, number: str, name: str) -> str:
        if cycle_id:
            return f"linear_cycles_json:{cycle_id}"
        return digest_source_id("linear_cycles_json", team, number, name)

    def _title(self, name: str, number: str, team: str) -> str:
        label = name or (f"Cycle {number}" if number else "Linear cycle")
        return f"{team} {label}" if team else label

    def _content(self, title: str, description: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (
            ("starts_at", "Starts"),
            ("ends_at", "Ends"),
            ("completed_at", "Completed"),
            ("progress", "Progress"),
            ("issue_count", "Issues"),
            ("scope", "Scope"),
            ("url", "URL"),
        ):
            if key in metadata:
                parts.append(f"{label}: {metadata[key]}")
        if description:
            parts.append(f"\n{description}")
        return "\n".join(parts)

    def _team(self, value: Any) -> dict[str, str]:
        if isinstance(value, dict):
            return {
                "key": self._text(value.get("key") or value.get("identifier")),
                "name": self._text(value.get("name")),
            }
        text = self._text(value)
        return {"key": text, "name": ""}

    def _get(self, record: dict[str, Any], *keys: str) -> Any:
        compact = {self._normalize_key(key): value for key, value in record.items()}
        for key in keys:
            value = record.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            if value is not None:
                return value
        return None

    def _normalize_key(self, value: str) -> str:
        return "".join(char for char in str(value).casefold() if char.isalnum())

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()

    def _bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        text = self._text(value).casefold()
        if not text:
            return None
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
        return None
