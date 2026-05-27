"""Adapter for GitHub commit JSON exports and API responses."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, ensure_utc, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GithubCommitsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_commits_json"

    @property
    def entity_types(self) -> list[str]:
        return ["commit"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self, *, since: SyncState | None = None, entity_types: list[str] | None = None
    ) -> IngestResult:
        result = IngestResult()
        if "commit" not in set(entity_types or self.entity_types):
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
            commits = parsed.get("commits")
            if isinstance(commits, list):
                return [item for item in commits if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        sha = self._text(record.get("sha") or record.get("node_id"))
        commit = record.get("commit") if isinstance(record.get("commit"), dict) else {}
        message = self._text(commit.get("message") or record.get("message"))
        title = message.splitlines()[0].strip() if message else sha
        html_url = self._text(record.get("html_url") or record.get("url"))
        author = self._person(commit.get("author") or record.get("author"))
        committer = self._person(commit.get("committer") or record.get("committer"))
        authored_at = parse_datetime(self._person_date(commit.get("author") or record.get("author")))
        committed_at = parse_datetime(self._person_date(commit.get("committer") or record.get("committer")))
        parents = self._parents(record.get("parents"))
        files = self._files(record.get("files"))
        stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
        if not sha and not message and not html_url:
            return None

        metadata = clean_metadata(
            {
                "external_id": sha,
                "sha": sha,
                "message": message,
                "html_url": html_url,
                "author": author,
                "committer": committer,
                "authored_at": authored_at.isoformat()
                if authored_at
                else self._person_date(commit.get("author") or record.get("author")),
                "committed_at": committed_at.isoformat()
                if committed_at
                else self._person_date(commit.get("committer") or record.get("committer")),
                "parents": parents,
                "files": files,
                "filenames": [file["filename"] for file in files if file.get("filename")],
                "stats": stats,
                "source_file": source_file,
                "record": record,
            }
        )
        now = datetime.now(timezone.utc)
        updated_at = committed_at or authored_at or now
        return KnowledgeUnit(
            source_project=SourceProject.GITHUB_COMMITS_JSON,
            source_id=f"github_commits_json:{sha}" if sha else html_url,
            source_entity_type="commit",
            title=title or "GitHub commit",
            content=message or title or html_url,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["github", "commit"],
            created_at=authored_at or updated_at,
            updated_at=updated_at,
        )

    def _person(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(
                value.get("name")
                or value.get("login")
                or value.get("username")
                or value.get("email")
            )
        return self._text(value)

    def _person_date(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("date") or value.get("timestamp"))
        return ""

    def _parents(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        parents: list[str] = []
        for item in value:
            parent = self._text(item.get("sha") if isinstance(item, dict) else item)
            if parent and parent not in parents:
                parents.append(parent)
        return parents

    def _files(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        files: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                files.append({"filename": item})
            elif isinstance(item, dict):
                files.append(
                    clean_metadata(
                        {
                            "filename": self._text(item.get("filename") or item.get("name")),
                            "status": self._text(item.get("status")),
                            "additions": item.get("additions"),
                            "deletions": item.get("deletions"),
                            "changes": item.get("changes"),
                            "previous_filename": self._text(item.get("previous_filename")),
                            "raw_url": self._text(item.get("raw_url")),
                            "blob_url": self._text(item.get("blob_url")),
                        }
                    )
                )
        return [file for file in files if file.get("filename")]

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
