"""Adapter for Linear issue JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class LinearIssuesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "linear_issues_json"

    @property
    def entity_types(self) -> list[str]:
        return ["issue"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types and "issue" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        by_id: dict[str, KnowledgeUnit] = {}
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
            if unit.metadata.get("issue_id"):
                by_id[str(unit.metadata["issue_id"])] = unit
            if unit.metadata.get("identifier"):
                by_id[str(unit.metadata["identifier"])] = unit
        for unit in result.units:
            parent = by_id.get(str(unit.metadata.get("parent_id") or ""))
            if parent:
                result.edges.append(self._edge(parent.source_id, unit.source_id, "parent", EdgeRelation.CONTAINS))
            for related in unit.metadata.get("related_issue_ids", []):
                target = by_id.get(str(related))
                if target:
                    result.edges.append(self._edge(unit.source_id, target.source_id, "related", EdgeRelation.RELATES_TO))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges = sorted({edge.id: edge for edge in result.edges}.values(), key=lambda edge: edge.id)
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("issues", "data"):
                if isinstance(parsed.get(key), list):
                    return [item for item in parsed[key] if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit(self, record: dict[str, Any]) -> KnowledgeUnit | None:
        issue_id = self._text(record.get("id"))
        identifier = self._text(record.get("identifier") or record.get("number"))
        title = self._text(record.get("title"))
        description = self._text(record.get("description"))
        if not issue_id and not identifier and not title:
            return None
        created = parse_datetime(record.get("createdAt") or record.get("created_at"))
        updated = parse_datetime(record.get("updatedAt") or record.get("updated_at")) or created
        started = parse_datetime(record.get("startedAt") or record.get("started_at"))
        triaged = parse_datetime(record.get("triagedAt") or record.get("triaged_at"))
        completed = parse_datetime(record.get("completedAt") or record.get("completed_at"))
        canceled = parse_datetime(record.get("canceledAt") or record.get("canceled_at"))
        archived = parse_datetime(record.get("archivedAt") or record.get("archived_at"))
        labels = self._names(record.get("labels"))
        team = self._name(record.get("team"))
        project = self._name(record.get("project"))
        parent_id = self._parent_id(record.get("parent"))
        lifecycle_metadata = self._lifecycle_metadata(
            created=created,
            updated=updated,
            started=started,
            triaged=triaged,
            completed=completed,
            canceled=canceled,
            archived=archived,
        )
        metadata = {
            "issue_id": issue_id,
            "identifier": identifier,
            "title": title,
            "description": description,
            "state": self._name(record.get("state")),
            "priority": record.get("priority"),
            "assignee": self._name(record.get("assignee")),
            "creator": self._name(record.get("creator")),
            "team": team,
            "project": project,
            "labels": labels,
            "url": self._text(record.get("url")),
            "created_at": created.isoformat() if created else self._text(record.get("createdAt")),
            "updated_at": updated.isoformat() if updated else self._text(record.get("updatedAt")),
            "completed_at": completed.isoformat() if completed else self._text(record.get("completedAt")),
            **lifecycle_metadata,
            "parent_id": parent_id,
            "related_issue_ids": [self._text(item) for item in record.get("relatedIssueIds", []) if self._text(item)],
            "source_file": record.get("_source_file"),
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.LINEAR_ISSUES_JSON,
            source_id=f"linear_issues_json:{issue_id or identifier}" if (issue_id or identifier) else digest_source_id("linear_issues_json", title, created),
            source_entity_type="issue",
            title=title or identifier or issue_id,
            content=self._content(title, description, metadata),
            content_type=ContentType.INSIGHT,
            metadata=clean_metadata(metadata),
            tags=list(dict.fromkeys(tag for tag in ["linear", "issue", team, project, *labels] if tag)),
            created_at=created or now,
            updated_at=updated or completed or created or now,
        )

    def _edge(self, source_id: str, target_id: str, kind: str, relation: EdgeRelation) -> KnowledgeEdge:
        digest = hashlib.sha256(f"{source_id}|{kind}|{target_id}".encode("utf-8")).hexdigest()[:24]
        return KnowledgeEdge(id=f"linear_issues_json:{kind}:{digest}", from_unit_id=source_id, to_unit_id=target_id, relation=relation, source=EdgeSource.SOURCE, metadata={"kind": kind, "source_project": SourceProject.LINEAR_ISSUES_JSON.value})

    def _content(self, title: str, description: str, metadata: dict[str, Any]) -> str:
        parts = [item for item in (title, description) if item]
        for key, label in (("identifier", "Identifier"), ("state", "State"), ("priority", "Priority"), ("url", "URL")):
            if metadata.get(key) not in ("", None):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)

    def _lifecycle_metadata(
        self,
        *,
        created: datetime | None,
        updated: datetime | None,
        started: datetime | None,
        triaged: datetime | None,
        completed: datetime | None,
        canceled: datetime | None,
        archived: datetime | None,
    ) -> dict[str, Any]:
        terminal = completed or canceled or archived
        metadata: dict[str, Any] = {
            "started_at": started.isoformat() if started else None,
            "triaged_at": triaged.isoformat() if triaged else None,
            "canceled_at": canceled.isoformat() if canceled else None,
            "archived_at": archived.isoformat() if archived else None,
        }
        if created and updated:
            metadata["age_days"] = self._days_between(created, updated)
        if created and triaged:
            metadata["time_to_triage_days"] = self._days_between(created, triaged)
        if created and started:
            metadata["time_to_start_days"] = self._days_between(created, started)
        if started and terminal:
            metadata["cycle_time_days"] = self._days_between(started, terminal)
        if created and terminal:
            metadata["lead_time_days"] = self._days_between(created, terminal)
        if terminal and updated:
            metadata["terminal_state_age_days"] = self._days_between(terminal, updated)
        return metadata

    def _days_between(self, start: datetime, end: datetime) -> int:
        return max(0, (end - start).days)

    def _parent_id(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("id") or value.get("identifier"))
        return self._text(value)

    def _names(self, value: Any) -> list[str]:
        if isinstance(value, dict) and isinstance(value.get("nodes"), list):
            value = value["nodes"]
        if not isinstance(value, list):
            return []
        names: list[str] = []
        for item in value:
            name = self._name(item)
            if name and name not in names:
                names.append(name)
        return names

    def _name(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("name") or value.get("displayName") or value.get("title") or value.get("id"))
        return self._text(value)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
