"""Adapter for Jira issue CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class JiraIssuesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "jira_issues_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["issue", "component", "fix_version"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        issues: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                issues.append(unit)
                components = list(unit.metadata.get("components") or [])
                if "issue" in allowed_types:
                    result.units.append(unit)
                    result.edges.extend(self._edges_for_unit(unit))
                if "component" in allowed_types:
                    result.units.extend(self._component_units(components, unit))
                if {"issue", "component"}.issubset(allowed_types):
                    result.edges.extend(self._component_edges(unit, components))
        fix_version_units = self._fix_version_units(issues) if "fix_version" in allowed_types else []
        result.units.extend(fix_version_units)
        if {"issue", "fix_version"}.issubset(allowed_types):
            result.edges.extend(self._fix_version_edges(issues, fix_version_units))
        result.units = list({unit.source_id: unit for unit in result.units}.values())
        result.edges = list({edge.id: edge for edge in result.edges}.values())
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.csv") if child.is_file())

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in csv.DictReader(handle)]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        key = self._first(row, "Issue key", "Key", "issue_key")
        summary = self._first(row, "Summary", "Title")
        description = self._first(row, "Description")
        if not key and not summary:
            return None
        created = self._parse_datetime(self._first(row, "Created", "Created date"))
        updated = self._parse_datetime(self._first(row, "Updated", "Updated date")) or created
        resolved = self._parse_datetime(self._first(row, "Resolved", "Resolution date"))
        labels = self._split(self._first(row, "Labels", "Label"))
        components = self._split(self._first(row, "Components", "Component/s", "Component"))
        fix_versions = self._split(self._first(row, "Fix versions", "Fix Version/s", "Fix version"))
        metadata = {
            "issue_key": key,
            "summary": summary,
            "description": description,
            "issue_type": self._first(row, "Issue Type", "Type"),
            "status": self._first(row, "Status"),
            "priority": self._first(row, "Priority"),
            "assignee": self._first(row, "Assignee"),
            "reporter": self._first(row, "Reporter"),
            "parent_key": self._first(row, "Parent key", "Parent"),
            "labels": labels,
            "components": components,
            "fix_versions": fix_versions,
            "created_at": created.isoformat() if created else self._first(row, "Created", "Created date"),
            "updated_at": updated.isoformat() if updated else self._first(row, "Updated", "Updated date"),
            "resolved_at": resolved.isoformat() if resolved else self._first(row, "Resolved", "Resolution date"),
            "source_file": source_file,
            "row": dict(row),
        }
        issue_links = self._issue_links(row)
        if issue_links:
            metadata["issue_links"] = issue_links
        now = datetime.now(timezone.utc)
        tags = ["jira", "issue", *labels, *[f"component:{item}" for item in components], *[f"fix_version:{item}" for item in fix_versions]]
        return KnowledgeUnit(
            source_project=SourceProject.JIRA_ISSUES_CSV,
            source_id=f"jira_issues_csv:{key}" if key else self._source_id(summary, description),
            source_entity_type="issue",
            title=summary or key,
            content=self._content(summary, description, metadata),
            content_type=ContentType.INSIGHT,
            metadata={item_key: value for item_key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(item for item in tags if item)),
            created_at=created or now,
            updated_at=updated or created or now,
        )

    def _edges_for_unit(self, unit: KnowledgeUnit) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for kind in ("assignee", "reporter", "parent_key"):
            value = unit.metadata.get(kind)
            if value:
                relation = EdgeRelation.REFERENCES if kind == "parent_key" else EdgeRelation.RELATES_TO
                target = self._issue_source_id(str(value)) if kind == "parent_key" else f"jira:{kind}:{value}"
                edges.append(self._edge(unit.source_id, target, relation, kind, str(value)))
        for link in unit.metadata.get("issue_links") or []:
            if not isinstance(link, dict):
                continue
            target_key = self._first(link, "target_key")
            kind = self._first(link, "kind")
            relation_name = self._first(link, "relation")
            relation = EdgeRelation.REFERENCES if relation_name == EdgeRelation.REFERENCES.value else EdgeRelation.RELATES_TO
            if target_key:
                edges.append(
                    self._edge(
                        unit.source_id,
                        self._issue_source_id(target_key),
                        relation,
                        kind or "issue_link",
                        target_key,
                    )
                )
        return edges

    def _component_units(self, components: list[str], issue: KnowledgeUnit) -> list[KnowledgeUnit]:
        return [
            KnowledgeUnit(
                source_project=SourceProject.JIRA_ISSUES_CSV,
                source_id=self._component_source_id(component),
                source_entity_type="component",
                title=component,
                content=f"Jira component: {component}",
                content_type=ContentType.METADATA,
                metadata={"name": component, "issue_source_ids": [issue.source_id]},
                tags=["jira", "component"],
                created_at=issue.created_at,
                updated_at=issue.updated_at,
            )
            for component in components
        ]

    def _component_edges(self, issue: KnowledgeUnit, components: list[str]) -> list[KnowledgeEdge]:
        return [
            self._edge(issue.source_id, self._component_source_id(component), EdgeRelation.RELATES_TO, "component", component)
            for component in components
        ]

    def _component_source_id(self, component: str) -> str:
        digest = hashlib.sha256(component.casefold().encode("utf-8")).hexdigest()[:24]
        return f"jira_issues_csv:component:{digest}"

    def _fix_version_units(self, issues: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for issue in issues:
            for version in issue.metadata.get("fix_versions") or []:
                key = self._normalized_fix_version(str(version))
                if not key:
                    continue
                grouped.setdefault(key, []).append(issue)
                names.setdefault(key, str(version).strip())

        units: list[KnowledgeUnit] = []
        for key, version_issues in sorted(grouped.items()):
            unique_issues = sorted({issue.source_id: issue for issue in version_issues}.values(), key=lambda issue: issue.source_id)
            created_dates = [issue.created_at for issue in unique_issues]
            updated_dates = [issue.updated_at for issue in unique_issues]
            name = names[key]
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.JIRA_ISSUES_CSV,
                    source_id=self._fix_version_source_id(key),
                    source_entity_type="fix_version",
                    title=name,
                    content=f"Jira fix version: {name}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "name": name,
                        "normalized_name": key,
                        "issue_count": len(unique_issues),
                        "statuses": sorted({str(issue.metadata.get("status")) for issue in unique_issues if issue.metadata.get("status")}),
                        "components": sorted({component for issue in unique_issues for component in (issue.metadata.get("components") or [])}),
                        "first_created_at": min(created_dates).isoformat() if created_dates else "",
                        "last_updated_at": max(updated_dates).isoformat() if updated_dates else "",
                        "issue_source_ids": [issue.source_id for issue in unique_issues],
                    },
                    tags=["jira", "fix_version"],
                    created_at=min(created_dates),
                    updated_at=max(updated_dates),
                )
            )
        return units

    def _fix_version_edges(self, issues: list[KnowledgeUnit], fix_versions: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        version_ids = {str(unit.metadata["normalized_name"]): unit.source_id for unit in fix_versions}
        edges: list[KnowledgeEdge] = []
        for issue in issues:
            for version in issue.metadata.get("fix_versions") or []:
                key = self._normalized_fix_version(str(version))
                target = version_ids.get(key)
                if target:
                    edges.append(self._edge(issue.source_id, target, EdgeRelation.RELATES_TO, "fix_version", str(version)))
        return edges

    def _fix_version_source_id(self, normalized_name: str) -> str:
        digest = hashlib.sha256(normalized_name.encode("utf-8")).hexdigest()[:24]
        return f"jira_issues_csv:fix_version:{digest}"

    def _normalized_fix_version(self, value: str) -> str:
        return " ".join(value.casefold().split())

    def _edge(self, source_id: str, target: str, relation: EdgeRelation, kind: str, value: str) -> KnowledgeEdge:
        digest = hashlib.sha256(f"{source_id}|{relation}|{target}".encode("utf-8")).hexdigest()[:24]
        return KnowledgeEdge(id=f"jira_issues_csv:{digest}", from_unit_id=source_id, to_unit_id=target, relation=relation, source=EdgeSource.SOURCE, metadata={"kind": kind, "value": value})

    def _issue_links(self, row: dict[str, Any]) -> list[dict[str, str]]:
        specs = (
            (("Blocks", "Blocked", "Outward issue link (Blocks)"), "blocks", EdgeRelation.RELATES_TO),
            (("Is blocked by", "Blocked by", "Inward issue link (Blocks)"), "is_blocked_by", EdgeRelation.RELATES_TO),
            (("Relates to", "Related", "Related issues", "Issue Links"), "relates_to", EdgeRelation.RELATES_TO),
            (("Duplicates", "Duplicate", "Outward issue link (Duplicate)"), "duplicates", EdgeRelation.RELATES_TO),
            (("Is duplicated by", "Duplicated by", "Inward issue link (Duplicate)"), "is_duplicated_by", EdgeRelation.RELATES_TO),
            (("Epic Link", "Epic", "Epic key"), "epic", EdgeRelation.REFERENCES),
        )
        links: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for columns, kind, relation in specs:
            for target_key in self._linked_issue_keys(self._first(row, *columns)):
                key = (kind, target_key)
                if key in seen:
                    continue
                seen.add(key)
                links.append(
                    {
                        "kind": kind,
                        "target_key": target_key,
                        "relation": relation.value,
                    }
                )
        return links

    def _linked_issue_keys(self, value: str) -> list[str]:
        keys: list[str] = []
        for item in re.split(r"[,;|\s]+", value or ""):
            text = item.strip().upper()
            if re.fullmatch(r"[A-Z][A-Z0-9]+-\d+", text) and text not in keys:
                keys.append(text)
        return keys

    def _content(self, summary: str, description: str, metadata: dict[str, Any]) -> str:
        parts = [item for item in (summary, description) if item]
        for key, label in (("issue_key", "Key"), ("issue_type", "Type"), ("status", "Status"), ("priority", "Priority")):
            if metadata.get(key):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)

    def _source_id(self, *parts: str) -> str:
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
        return f"jira_issues_csv:{digest}"

    def _issue_source_id(self, key: str) -> str:
        return f"jira_issues_csv:{key.strip().upper()}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _split(self, value: str) -> list[str]:
        items: list[str] = []
        for item in re.split(r"[,;|]", value or ""):
            text = item.strip()
            if text and text not in items:
                items.append(text)
        return items

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _parse_datetime(self, value: Any) -> datetime | None:
        text = "" if value is None else str(value).strip()
        if not text:
            return None
        for candidate in (text, text.replace("Z", "+00:00")):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate))
            except ValueError:
                pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
