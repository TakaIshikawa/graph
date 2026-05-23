"""Adapter for GitHub pull request JSON exports and API responses."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from graph.adapters._personal_exports import (
    clean_metadata,
    digest_source_id,
    ensure_utc,
    iter_paths,
    parse_datetime,
    parse_int,
)
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GithubPullRequestsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_pull_requests_json"

    @property
    def entity_types(self) -> list[str]:
        return ["pull_request"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self, *, since: SyncState | None = None, entity_types: list[str] | None = None
    ) -> IngestResult:
        result = IngestResult()
        if "pull_request" not in set(entity_types or self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
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
        result.units = sorted(
            {unit.source_id: unit for unit in result.units}.values(),
            key=lambda unit: unit.source_id,
        )
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("items", "nodes", "pull_requests"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = self._text(record.get("title"))
        body = self._text(record.get("body") or record.get("bodyText"))
        number = parse_int(record.get("number"))
        state = self._text(record.get("state"))
        author = self._person(record.get("author") or record.get("user"))
        repository = self._repository(record)
        url = self._text(record.get("html_url") or record.get("url"))
        labels = self._labels(record.get("labels"))
        created_at = parse_datetime(record.get("created_at") or record.get("createdAt"))
        updated_at = (
            parse_datetime(record.get("updated_at") or record.get("updatedAt")) or created_at
        )
        merged_at = parse_datetime(record.get("merged_at") or record.get("mergedAt"))
        if not title and number is None and not url:
            return None

        metadata = clean_metadata(
            {
                "title": title,
                "number": number,
                "state": state,
                "author": author,
                "repository": repository,
                "url": url,
                "created_at": created_at.isoformat()
                if created_at
                else self._text(record.get("created_at") or record.get("createdAt")),
                "updated_at": updated_at.isoformat()
                if updated_at
                else self._text(record.get("updated_at") or record.get("updatedAt")),
                "merged_at": merged_at.isoformat()
                if merged_at
                else self._text(record.get("merged_at") or record.get("mergedAt")),
                "labels": labels,
                "body": body,
                "source_file": source_file,
                "record": record,
            }
        )
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.GITHUB_PULL_REQUESTS_JSON,
            source_id=self._source_id(repository, number, url, title),
            source_entity_type="pull_request",
            title=title or (f"GitHub pull request #{number}" if number is not None else url),
            content=self._content(title, body, state, repository, number, url, labels),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(["github", "pull_request", *labels])),
            created_at=created_at or now,
            updated_at=updated_at or merged_at or created_at or now,
        )

    def _source_id(self, repository: str, number: int | None, url: str, title: str) -> str:
        if repository and number is not None:
            return f"github_pull_requests_json:{repository}#{number}"
        return digest_source_id("github_pull_requests_json", url or repository, number, title)

    def _content(
        self,
        title: str,
        body: str,
        state: str,
        repository: str,
        number: int | None,
        url: str,
        labels: list[str],
    ) -> str:
        parts = [
            title,
            f"Repository: {repository}" if repository else "",
            f"Number: {number}" if number is not None else "",
            f"State: {state}" if state else "",
            f"Labels: {', '.join(labels)}" if labels else "",
            f"URL: {url}" if url else "",
            body,
        ]
        return "\n".join(part for part in parts if part)

    def _repository(self, record: dict[str, Any]) -> str:
        repository = record.get("repository")
        if repository is None and isinstance(record.get("base"), dict):
            repository = record["base"].get("repo")
        if isinstance(repository, dict):
            return self._text(
                repository.get("full_name")
                or repository.get("nameWithOwner")
                or repository.get("path_with_namespace")
                or repository.get("name")
            )
        return self._text(
            record.get("repository_full_name") or record.get("repositoryFullName") or repository
        )

    def _labels(self, value: Any) -> list[str]:
        if isinstance(value, dict):
            value = value.get("nodes") or value.get("items") or value.get("edges")
        if isinstance(value, str):
            items: list[Any] = value.split(",")
        elif isinstance(value, list):
            items = value
        else:
            items = []
        labels: list[str] = []
        for item in items:
            label = ""
            if isinstance(item, dict):
                node = item.get("node") if isinstance(item.get("node"), dict) else item
                label = self._text(node.get("name"))
            else:
                label = self._text(item)
            label = label.casefold()
            if label and label not in labels:
                labels.append(label)
        return labels

    def _person(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("login") or value.get("username") or value.get("name"))
        return self._text(value)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
