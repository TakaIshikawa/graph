"""Adapter for GitHub issue JSON and JSONL exports."""

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


URL_RE = re.compile(r"https?://[^\s<>)\]}\"']+")


class GithubIssuesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_issues_json"

    @property
    def entity_types(self) -> list[str]:
        return ["issue", "pull_request", "label", "milestone"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types) if entity_types is not None else {"issue", "pull_request"}
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
                if unit.source_entity_type not in requested:
                    continue
                result.units.append(unit)
                result.edges.extend(self._edges_from_unit(unit))
        label_units = self._label_units(issue_units)
        milestone_units = self._milestone_units(issue_units)
        if "label" in requested:
            result.units.extend(label_units)
        if "milestone" in requested:
            result.units.extend(milestone_units)
        if "label" in requested and requested.intersection({"issue", "pull_request"}):
            label_ids = {unit.metadata["label"]: unit.source_id for unit in label_units}
            for issue in issue_units:
                if issue.source_entity_type not in requested:
                    continue
                for label in issue.metadata.get("labels", []):
                    if label in label_ids:
                        result.edges.append(self._edge(issue.source_id, label_ids[label], "label", label, EdgeRelation.RELATES_TO))
        if "milestone" in requested and requested.intersection({"issue", "pull_request"}):
            milestone_ids = {unit.metadata["milestone_title"]: unit.source_id for unit in milestone_units}
            for issue in issue_units:
                if issue.source_entity_type not in requested:
                    continue
                milestone_title = self._text(issue.metadata.get("milestone_title"))
                milestone_id = milestone_ids.get(milestone_title)
                if milestone_id:
                    result.edges.append(self._milestone_edge(issue, milestone_id))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges = sorted({edge.id: edge for edge in result.edges}.values(), key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*") if child.is_file() and child.suffix.lower() in {".json", ".jsonl", ".ndjson"})

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            return [record for record in (json.loads(line) for line in text.splitlines() if line.strip()) if isinstance(record, dict)]
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [record for record in parsed if isinstance(record, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
            return [record for record in parsed["items"] if isinstance(record, dict)]
        return [parsed] if isinstance(parsed, dict) else []

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = self._text(record.get("title"))
        body = self._text(record.get("body"))
        number = self._parse_int(record.get("number"))
        url = self._text(record.get("html_url") or record.get("url"))
        repository = self._repository(record)
        if not title and not body and number is None:
            return None
        entity_type = "pull_request" if isinstance(record.get("pull_request"), dict) else "issue"
        created_at = self._parse_datetime(record.get("created_at"))
        updated_at = self._parse_datetime(record.get("updated_at")) or created_at
        closed_at = self._parse_datetime(record.get("closed_at"))
        labels = self._labels(record.get("labels"))
        author = self._author(record.get("user"))
        assignees = self._assignees(record)
        milestone = self._milestone_metadata(record.get("milestone"))
        metadata = {
            "title": title,
            "body": body,
            "state": self._text(record.get("state")),
            "labels": labels,
            "repository": repository,
            "author": author,
            "assignees": assignees,
            "url": url,
            "issue_number": number,
            "created_at": created_at.isoformat() if created_at else self._text(record.get("created_at")),
            "updated_at": updated_at.isoformat() if updated_at else self._text(record.get("updated_at")),
            "closed_at": closed_at.isoformat() if closed_at else self._text(record.get("closed_at")),
            **milestone,
            "pull_request": record.get("pull_request") if entity_type == "pull_request" else None,
            "source_file": source_file,
            "record": record,
        }
        now = datetime.now(timezone.utc)
        tags = ["github", entity_type, *labels, milestone.get("milestone_title")]
        return KnowledgeUnit(
            source_project=SourceProject.GITHUB_ISSUES_JSON,
            source_id=self._source_id(repository, number, url, title),
            source_entity_type=entity_type,
            title=title or f"GitHub #{number}",
            content=self._content(title, body, metadata["state"], repository, number, url, labels),
            content_type=ContentType.ARTIFACT if entity_type == "pull_request" else ContentType.INSIGHT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(tag for tag in tags if tag)),
            created_at=created_at or now,
            updated_at=updated_at or created_at or now,
        )

    def _source_id(self, repository: str, number: int | None, url: str, title: str) -> str:
        if repository and number is not None:
            return f"github_issues_json:{repository}#{number}"
        raw = url or f"{repository}|{number}|{title}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"github_issues_json:{digest}"

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
            repositories = sorted({repo for issue in linked_issues if (repo := self._text(issue.metadata.get("repository")))})
            metadata = {
                "label": label,
                "issue_source_ids": source_ids,
                "issue_count": len(source_ids),
                "repositories": repositories,
                "latest_updated_at": updated_at.isoformat(),
            }
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GITHUB_ISSUES_JSON,
                    source_id=self._label_source_id(label),
                    source_entity_type="label",
                    title=f"GitHub label: {label}",
                    content=f"GitHub label: {label}\nIssues: {len(source_ids)}",
                    content_type=ContentType.METADATA,
                    metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
                    tags=["github", "label", label],
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return units

    def _label_source_id(self, label: str) -> str:
        digest = hashlib.sha256(label.casefold().encode("utf-8")).hexdigest()[:24]
        return f"github_issues_json:label:{digest}"

    def _milestone_units(self, issues: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        metadata_by_title: dict[str, dict[str, Any]] = {}
        for issue in issues:
            title = self._text(issue.metadata.get("milestone_title"))
            if not title:
                continue
            grouped.setdefault(title, []).append(issue)
            metadata_by_title.setdefault(
                title,
                {
                    "milestone_title": title,
                    "milestone_state": self._text(issue.metadata.get("milestone_state")),
                    "milestone_due_on": self._text(issue.metadata.get("milestone_due_on")),
                    "milestone_number": issue.metadata.get("milestone_number"),
                },
            )

        units: list[KnowledgeUnit] = []
        now = datetime.now(timezone.utc)
        for title, linked_issues in grouped.items():
            created_at = min((issue.created_at for issue in linked_issues), default=now)
            updated_at = max((issue.updated_at for issue in linked_issues), default=created_at)
            source_ids = sorted({issue.source_id for issue in linked_issues})
            repositories = sorted({repo for issue in linked_issues if (repo := self._text(issue.metadata.get("repository")))})
            states = sorted({state for issue in linked_issues if (state := self._text(issue.metadata.get("milestone_state")))})
            due_dates = sorted({due for issue in linked_issues if (due := self._text(issue.metadata.get("milestone_due_on")))})
            numbers = sorted({number for issue in linked_issues if (number := issue.metadata.get("milestone_number")) is not None})
            milestone_metadata = metadata_by_title[title]
            metadata = {
                **milestone_metadata,
                "milestone_title": title,
                "issue_source_ids": source_ids,
                "issue_count": len(source_ids),
                "repositories": repositories,
                "states": states,
                "due_dates": due_dates,
                "numbers": numbers,
                "latest_updated_at": updated_at.isoformat(),
            }
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GITHUB_ISSUES_JSON,
                    source_id=self._milestone_source_id(title),
                    source_entity_type="milestone",
                    title=f"GitHub milestone: {title}",
                    content=f"GitHub milestone: {title}\nIssues: {len(source_ids)}",
                    content_type=ContentType.METADATA,
                    metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
                    tags=["github", "milestone", title],
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return units

    def _milestone_source_id(self, title: str) -> str:
        digest = hashlib.sha256(title.casefold().encode("utf-8")).hexdigest()[:24]
        return f"github_issues_json:milestone:{digest}"

    def _edges_from_unit(self, unit: KnowledgeUnit) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        metadata = unit.metadata
        for kind, value in (("author", metadata.get("author")),):
            text = self._text(value)
            if text:
                edges.append(self._edge(unit.source_id, self._target_id(kind, text), kind, text, EdgeRelation.RELATES_TO))
        for assignee in metadata.get("assignees", []):
            text = self._text(assignee)
            if text:
                edges.append(self._edge(unit.source_id, self._target_id("assignee", text), "assignee", text, EdgeRelation.RELATES_TO))
        for url in self._mentioned_urls(metadata.get("body", "")):
            edges.append(self._edge(unit.source_id, self._target_id("url", url), "mentioned_url", url, EdgeRelation.REFERENCES))
        return edges

    def _milestone_edge(self, unit: KnowledgeUnit, milestone_id: str) -> KnowledgeEdge:
        title = self._text(unit.metadata.get("milestone_title"))
        edge = self._edge(unit.source_id, milestone_id, "milestone", title, EdgeRelation.RELATES_TO)
        edge.metadata.update(
            {
                "milestone_title": title,
                "milestone_state": self._text(unit.metadata.get("milestone_state")),
                "milestone_due_on": self._text(unit.metadata.get("milestone_due_on")),
                "milestone_number": unit.metadata.get("milestone_number"),
                "from_entity_type": unit.source_entity_type,
                "to_entity_type": "milestone",
            }
        )
        edge.metadata = {key: value for key, value in edge.metadata.items() if value not in ("", None, [])}
        return edge

    def _edge(self, source_id: str, target_id: str, kind: str, value: str, relation: EdgeRelation) -> KnowledgeEdge:
        digest = hashlib.sha256(f"{source_id}|{kind}|{target_id}".encode("utf-8")).hexdigest()[:24]
        return KnowledgeEdge(
            id=f"github_issues_json:{kind}:{digest}",
            from_unit_id=source_id,
            to_unit_id=target_id,
            relation=relation,
            source=EdgeSource.SOURCE,
            metadata={
                "kind": kind,
                "value": value,
                "source_project": SourceProject.GITHUB_ISSUES_JSON.value,
            },
        )

    def _target_id(self, kind: str, value: str) -> str:
        normalized = value.strip().casefold() if kind != "url" else value.strip()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
        prefix = "url" if kind == "url" else kind
        return f"github:{prefix}:{digest}"

    def _mentioned_urls(self, text: Any) -> list[str]:
        urls: list[str] = []
        for match in URL_RE.findall(self._text(text)):
            url = match.rstrip(".,;:")
            if url and url not in urls:
                urls.append(url)
        return urls

    def _content(self, title: str, body: str, state: str, repository: str, number: int | None, url: str, labels: list[str]) -> str:
        parts = [title] if title else []
        if body:
            parts.append(body)
        for label, value in (("Repository", repository), ("Number", number), ("State", state), ("URL", url)):
            if value not in ("", None):
                parts.append(f"{label}: {value}")
        if labels:
            parts.append(f"Labels: {', '.join(labels)}")
        return "\n".join(parts)

    def _repository(self, record: dict[str, Any]) -> str:
        repository = record.get("repository")
        if isinstance(repository, dict):
            return self._text(repository.get("full_name") or repository.get("name"))
        return self._text(record.get("repository") or record.get("repo") or record.get("repository_full_name"))

    def _labels(self, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            raw = value.replace(";", ",").replace("|", ",").split(",")
        else:
            raw = [item.get("name") if isinstance(item, dict) else item for item in value]
        labels: list[str] = []
        for item in raw:
            label = self._text(item).lower()
            if label and label not in labels:
                labels.append(label)
        return labels

    def _author(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("login") or value.get("name"))
        return self._text(value)

    def _assignees(self, record: dict[str, Any]) -> list[str]:
        raw = record.get("assignees")
        if raw is None and record.get("assignee") is not None:
            raw = [record.get("assignee")]
        if not isinstance(raw, list):
            return []
        assignees: list[str] = []
        for item in raw:
            name = self._author(item)
            if name and name not in assignees:
                assignees.append(name)
        return assignees

    def _milestone_metadata(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        due_on = self._parse_datetime(value.get("due_on"))
        return {
            "milestone_title": self._text(value.get("title")),
            "milestone_state": self._text(value.get("state")),
            "milestone_due_on": due_on.isoformat() if due_on else self._text(value.get("due_on")),
            "milestone_number": self._parse_int(value.get("number")),
        }

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
