"""Adapter for GitHub notifications JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GithubNotificationsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_notifications_json"

    @property
    def entity_types(self) -> list[str]:
        return ["notification"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "notification" not in set(entity_types or self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
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
                result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
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
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("notifications", "items", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        subject = record.get("subject") if isinstance(record.get("subject"), dict) else {}
        repository = record.get("repository") if isinstance(record.get("repository"), dict) else {}
        notification_id = self._text(record.get("id"))
        repository_full_name = self._text(repository.get("full_name") or record.get("repository_full_name") or record.get("repo"))
        subject_title = self._text(subject.get("title") or record.get("subject_title") or record.get("title"))
        subject_type = self._text(subject.get("type") or record.get("subject_type"))
        subject_url = self._text(subject.get("url") or subject.get("latest_comment_url") or record.get("subject_url"))
        reason = self._text(record.get("reason"))
        unread = self._parse_bool(record.get("unread"))
        url = self._text(record.get("url") or record.get("html_url") or subject_url)
        updated_at = self._parse_datetime(record.get("updated_at"))
        last_read_at = self._parse_datetime(record.get("last_read_at"))
        subscription_url = self._text(record.get("subscription_url"))
        if not notification_id and not subject_title and not url:
            return None
        metadata = {
            "id": notification_id,
            "repository": repository_full_name,
            "subject_title": subject_title,
            "subject_type": subject_type,
            "subject_url": subject_url,
            "reason": reason,
            "unread": unread,
            "url": url,
            "updated_at": updated_at.isoformat() if updated_at else self._text(record.get("updated_at")),
            "last_read_at": last_read_at.isoformat() if last_read_at else self._text(record.get("last_read_at")),
            "subscription_url": subscription_url,
            "source_file": source_file,
            "record": record,
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.GITHUB_NOTIFICATIONS_JSON,
            source_id=self._source_id(notification_id, url, subject_title),
            source_entity_type="notification",
            title=subject_title or url or notification_id,
            content=self._content(repository_full_name, subject_title, subject_type, reason, url),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(item for item in ["github", "notification", repository_full_name, subject_type] if item)),
            created_at=updated_at or now,
            updated_at=updated_at or now,
        )

    def _content(self, repository: str, title: str, subject_type: str, reason: str, url: str) -> str:
        parts = [title, f"Repository: {repository}" if repository else "", f"Type: {subject_type}" if subject_type else "", f"Reason: {reason}" if reason else "", f"URL: {url}" if url else ""]
        return "\n".join(part for part in parts if part)

    def _source_id(self, notification_id: str, url: str, title: str) -> str:
        digest = hashlib.sha256((notification_id or url or title).encode("utf-8")).hexdigest()[:24]
        return f"github_notifications_json:{digest}"

    def _parse_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        text = self._text(value).casefold()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
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
