"""Adapter for Medium bookmarks JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class MediumBookmarksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "medium_bookmarks_json"

    @property
    def entity_types(self) -> list[str]:
        return ["article_bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "article_bookmark" not in set(entity_types or self.entity_types):
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
            for key in ("bookmarks", "posts", "articles", "items", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = self._text(record.get("title") or record.get("name"))
        subtitle = self._text(record.get("subtitle") or record.get("description"))
        author = self._person(record.get("author") or record.get("creator"))
        publication = self._text(record.get("publication") or record.get("publication_title"))
        url = self._text(record.get("url") or record.get("canonical_url") or record.get("medium_url"))
        tags = self._list(record.get("tags") or record.get("topics"))
        claps = self._parse_int(record.get("claps") or record.get("clap_count"))
        responses = self._parse_int(record.get("responses") or record.get("responses_count") or record.get("comments_count"))
        saved_at = self._parse_datetime(record.get("saved_at") or record.get("bookmarked_at") or record.get("created_at"))
        published_at = self._parse_datetime(record.get("published_at") or record.get("first_published_at"))
        updated_at = self._parse_datetime(record.get("updated_at") or record.get("last_modified_at")) or saved_at or published_at
        if not title and not url:
            return None
        metadata = {
            "title": title,
            "subtitle": subtitle,
            "author": author,
            "publication": publication,
            "url": url,
            "tags": tags,
            "claps": claps,
            "responses": responses,
            "saved_at": saved_at.isoformat() if saved_at else self._text(record.get("saved_at")),
            "published_at": published_at.isoformat() if published_at else self._text(record.get("published_at")),
            "updated_at": updated_at.isoformat() if updated_at else self._text(record.get("updated_at")),
            "source_file": source_file,
            "record": record,
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.MEDIUM_BOOKMARKS_JSON,
            source_id=self._source_id(url, title, author),
            source_entity_type="article_bookmark",
            title=title or url,
            content=self._content(title, subtitle, author, publication, url),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(["medium", "article_bookmark", *tags])),
            created_at=published_at or saved_at or now,
            updated_at=updated_at or now,
        )

    def _content(self, title: str, subtitle: str, author: str, publication: str, url: str) -> str:
        parts = [title, subtitle, f"Author: {author}" if author else "", f"Publication: {publication}" if publication else "", f"URL: {url}" if url else ""]
        return "\n".join(part for part in parts if part)

    def _source_id(self, url: str, title: str, author: str) -> str:
        digest = hashlib.sha256((url or f"{title}|{author}").encode("utf-8")).hexdigest()[:24]
        return f"medium_bookmarks_json:{digest}"

    def _list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [self._text(item.get("name") if isinstance(item, dict) else item) for item in value if self._text(item.get("name") if isinstance(item, dict) else item)]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return []

    def _person(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("name") or value.get("username"))
        return self._text(value)

    def _parse_int(self, value: Any) -> int | None:
        if value in ("", None):
            return None
        try:
            return int(float(str(value).strip()))
        except ValueError:
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
