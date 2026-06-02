"""Adapter for GitHub pull request review comments JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime, parse_int
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GithubReviewCommentsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_review_comments_json"

    @property
    def entity_types(self) -> list[str]:
        return ["review_comment"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "review_comment" not in entity_types:
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
            for key in ("review_comments", "comments", "data", "nodes"):
                records = self._records(value.get(key))
                if records:
                    return records
            return [value]
        return []

    def _unit(self, record: dict[str, Any]) -> KnowledgeUnit | None:
        comment_id = self._text(self._get(record, "id", "comment_id"))
        repo = self._repo(record)
        pr = parse_int(self._get(record, "pull_request_number", "pullRequestNumber", "pr_number", "number"))
        path = self._text(self._get(record, "path", "file_path", "filePath"))
        body = self._text(self._get(record, "body", "comment", "text"))
        author = self._person(record.get("user") or record.get("author"))
        if not any([comment_id, repo, pr, path, body]):
            return None
        created = parse_datetime(self._get(record, "created_at", "createdAt"))
        updated = parse_datetime(self._get(record, "updated_at", "updatedAt")) or created
        url = self._text(self._get(record, "html_url", "htmlUrl", "url"))
        metadata = clean_metadata({"comment_id": comment_id, "repository": repo, "pull_request_number": pr, "path": path, "position": parse_int(self._get(record, "position", "original_position")), "line": parse_int(self._get(record, "line", "original_line")), "author": author, "body": body, "url": url, "api_url": self._text(self._get(record, "pull_request_review_id", "review_id")), "created_at": created.isoformat() if created else self._text(self._get(record, "created_at", "createdAt")), "updated_at": updated.isoformat() if updated else self._text(self._get(record, "updated_at", "updatedAt")), "resolved": self._bool(self._get(record, "resolved", "is_resolved")), "source_file": record.get("_source_file")})
        now = datetime.now(timezone.utc)
        title = f"GitHub review comment {repo}#{pr}" if repo and pr else "GitHub review comment"
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{comment_id}" if comment_id else digest_source_id(self.name, repo, pr, path, body), source_entity_type="review_comment", title=title, content="\n".join(part for part in [title, body, f"Path: {path}" if path else "", f"Author: {author}" if author else "", f"URL: {url}" if url else ""] if part), content_type=ContentType.METADATA, metadata=metadata, tags=list(dict.fromkeys(tag for tag in ["github", "review_comment", repo] if tag)), created_at=created or now, updated_at=updated or created or now)

    def _repo(self, record: dict[str, Any]) -> str:
        repo = record.get("repository") or record.get("repo")
        if isinstance(repo, dict):
            return self._text(repo.get("full_name") or repo.get("nameWithOwner") or repo.get("name"))
        return self._text(self._get(record, "repository", "repo", "repo_name", "repository_full_name"))

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
            return self._text(value.get("login") or value.get("name"))
        return self._text(value)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()

    def _bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        text = self._text(value).casefold()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
        return None
