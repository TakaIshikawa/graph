"""Adapter for GitLab issue JSON exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class GitlabIssuesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "gitlab_issues_json"

    @property
    def entity_types(self) -> list[str]:
        return ["issue", "label"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types) if entity_types is not None else {"issue"}
        if not requested.intersection(self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        issue_units: list[KnowledgeUnit] = []
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
                issue_units.append(unit)
                if "issue" in requested:
                    result.units.append(unit)
                    result.edges.extend(self._edges_for_record(unit, record))
        label_units = self._label_units(issue_units)
        if "label" in requested:
            result.units.extend(label_units)
        if {"issue", "label"}.issubset(requested):
            label_ids = {unit.metadata["label"]: unit.source_id for unit in label_units}
            for issue in issue_units:
                for label in issue.metadata.get("labels", []):
                    if label in label_ids:
                        result.edges.append(self._edge(issue.source_id, label_ids[label], EdgeRelation.RELATES_TO, "label", label))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges = sorted({edge.id: edge for edge in result.edges}.values(), key=lambda edge: edge.id)
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
            return [record for record in parsed if isinstance(record, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("issues"), list):
            return [record for record in parsed["issues"] if isinstance(record, dict)]
        return [parsed] if isinstance(parsed, dict) else []

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = self._text(record.get("title"))
        body = self._text(record.get("description") or record.get("body"))
        iid = self._parse_int(record.get("iid") or record.get("issue_iid"))
        url = self._text(record.get("web_url") or record.get("url"))
        project_path = self._project_path(record)
        if not title and not body and iid is None and not url:
            return None
        created_at = self._parse_datetime(record.get("created_at"))
        updated_at = self._parse_datetime(record.get("updated_at")) or created_at
        closed_at = self._parse_datetime(record.get("closed_at"))
        labels = self._labels(record.get("labels"))
        author = self._person(record.get("author"))
        assignees = [self._person(item) for item in self._as_list(record.get("assignees"))]
        if not assignees:
            assignee = self._person(record.get("assignee"))
            assignees = [assignee] if assignee else []
        milestone = self._milestone(record.get("milestone"))
        metadata = {
            "title": title,
            "body": body,
            "state": self._text(record.get("state")),
            "labels": labels,
            "project_path": project_path,
            "iid": iid,
            "author": author,
            "assignees": [item for item in assignees if item],
            "milestone": milestone,
            "web_url": url,
            "referenced_urls": self._referenced_urls(body),
            "created_at": created_at.isoformat() if created_at else self._text(record.get("created_at")),
            "updated_at": updated_at.isoformat() if updated_at else self._text(record.get("updated_at")),
            "closed_at": closed_at.isoformat() if closed_at else self._text(record.get("closed_at")),
            "source_file": source_file,
            "record": record,
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.GITLAB_ISSUES_JSON,
            source_id=self._source_id(project_path, iid, url, title),
            source_entity_type="issue",
            title=title or f"GitLab issue {iid}",
            content=self._content(title, body, metadata["state"], project_path, iid, url, labels),
            content_type=ContentType.INSIGHT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(["gitlab", "issue", *labels])),
            created_at=created_at or now,
            updated_at=updated_at or created_at or now,
        )

    def _edges_for_record(self, unit: KnowledgeUnit, record: dict[str, Any]) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for role, value in (("author", unit.metadata.get("author")),):
            if value:
                edges.append(self._edge(unit.source_id, f"gitlab:{role}:{value}", EdgeRelation.RELATES_TO, role, value))
        for assignee in unit.metadata.get("assignees", []):
            edges.append(self._edge(unit.source_id, f"gitlab:assignee:{assignee}", EdgeRelation.RELATES_TO, "assignee", assignee))
        milestone = unit.metadata.get("milestone")
        if milestone:
            edges.append(self._edge(unit.source_id, f"gitlab:milestone:{milestone}", EdgeRelation.RELATES_TO, "milestone", milestone))
        for url in [unit.metadata.get("web_url"), *unit.metadata.get("referenced_urls", [])]:
            if url:
                edges.append(self._edge(unit.source_id, f"url:{url}", EdgeRelation.REFERENCES, "url", url))
        return list({edge.id: edge for edge in edges}.values())

    def _edge(self, source_id: str, target: str, relation: EdgeRelation, kind: str, value: str) -> KnowledgeEdge:
        digest = hashlib.sha256(f"{source_id}|{relation}|{target}".encode("utf-8")).hexdigest()[:24]
        return KnowledgeEdge(
            id=f"gitlab_issues_json:{digest}",
            from_unit_id=source_id,
            to_unit_id=target,
            relation=relation,
            source=EdgeSource.SOURCE,
            metadata={"kind": kind, "value": value},
        )

    def _label_units(self, issues: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for issue in issues:
            for label in issue.metadata.get("labels", []):
                grouped.setdefault(label, []).append(issue)

        units: list[KnowledgeUnit] = []
        now = datetime.now(timezone.utc)
        for label, linked_issues in grouped.items():
            created_at = min((issue.created_at for issue in linked_issues), default=now)
            updated_at = max((issue.updated_at for issue in linked_issues), default=created_at)
            source_ids = sorted({issue.source_id for issue in linked_issues})
            project_paths = sorted({path for issue in linked_issues if (path := self._text(issue.metadata.get("project_path")))})
            metadata = {
                "label": label,
                "issue_source_ids": source_ids,
                "issue_count": len(source_ids),
                "project_paths": project_paths,
                "latest_updated_at": updated_at.isoformat(),
            }
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GITLAB_ISSUES_JSON,
                    source_id=self._label_source_id(label),
                    source_entity_type="label",
                    title=f"GitLab label: {label}",
                    content=f"GitLab label: {label}\nIssues: {len(source_ids)}",
                    content_type=ContentType.METADATA,
                    metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
                    tags=["gitlab", "label", label],
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return units

    def _label_source_id(self, label: str) -> str:
        digest = hashlib.sha256(label.casefold().encode("utf-8")).hexdigest()[:24]
        return f"gitlab_issues_json:label:{digest}"

    def _project_path(self, record: dict[str, Any]) -> str:
        project = record.get("project")
        if isinstance(project, dict):
            return self._text(project.get("path_with_namespace") or project.get("full_path") or project.get("name"))
        return self._text(record.get("project_path") or record.get("project_full_path") or record.get("path_with_namespace"))

    def _source_id(self, project_path: str, iid: int | None, url: str, title: str) -> str:
        if project_path and iid is not None:
            return f"gitlab_issues_json:{project_path}#{iid}"
        raw = url or f"{project_path}|{iid}|{title}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"gitlab_issues_json:{digest}"

    def _content(self, title: str, body: str, state: str, project_path: str, iid: int | None, url: str, labels: list[str]) -> str:
        parts = [item for item in (title, body) if item]
        for label, value in (("Project", project_path), ("IID", iid), ("State", state), ("URL", url)):
            if value not in ("", None):
                parts.append(f"{label}: {value}")
        if labels:
            parts.append(f"Labels: {', '.join(labels)}")
        return "\n".join(parts)

    def _labels(self, value: Any) -> list[str]:
        raw = value.replace(";", ",").split(",") if isinstance(value, str) else self._as_list(value)
        labels: list[str] = []
        for item in raw:
            label = self._text(item.get("name") if isinstance(item, dict) else item).lower()
            if label and label not in labels:
                labels.append(label)
        return labels

    def _person(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("username") or value.get("login") or value.get("name"))
        return self._text(value)

    def _milestone(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("title") or value.get("name"))
        return self._text(value)

    def _referenced_urls(self, text: str) -> list[str]:
        urls: list[str] = []
        for match in re.findall(r'https?://[^\s<>)\]"]+', text or ""):
            url = match.rstrip(".,;:")
            if url and url not in urls:
                urls.append(url)
        return urls

    def _as_list(self, value: Any) -> list[Any]:
        return value if isinstance(value, list) else ([] if value in (None, "") else [value])

    def _parse_int(self, value: Any) -> int | None:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

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
