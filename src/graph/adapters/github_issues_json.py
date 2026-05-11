"""Adapter for GitHub issue JSON and JSONL exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GithubIssuesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_issues_json"

    @property
    def entity_types(self) -> list[str]:
        return ["issue", "pull_request"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types or self.entity_types)
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for record in records:
                unit = self._unit_from_record(record, path.name)
                if unit is None or unit.source_entity_type not in requested:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
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
        metadata = {
            "title": title,
            "body": body,
            "state": self._text(record.get("state")),
            "labels": labels,
            "repository": repository,
            "author": author,
            "url": url,
            "issue_number": number,
            "created_at": created_at.isoformat() if created_at else self._text(record.get("created_at")),
            "updated_at": updated_at.isoformat() if updated_at else self._text(record.get("updated_at")),
            "closed_at": closed_at.isoformat() if closed_at else self._text(record.get("closed_at")),
            "pull_request": record.get("pull_request") if entity_type == "pull_request" else None,
            "source_file": source_file,
            "record": record,
        }
        now = datetime.now(timezone.utc)
        tags = ["github", entity_type, *labels]
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
