"""Adapter for Product Hunt bookmarks JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class ProductHuntBookmarksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "product_hunt_bookmarks_json"

    @property
    def entity_types(self) -> list[str]:
        return ["product_bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "product_bookmark" not in set(entity_types or self.entity_types):
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
            for key in ("bookmarks", "posts", "products", "items", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        name = self._text(record.get("name") or record.get("product_name") or record.get("title"))
        tagline = self._text(record.get("tagline") or record.get("subtitle"))
        description = self._text(record.get("description"))
        url = self._text(record.get("url") or record.get("product_url") or record.get("discussion_url"))
        makers = self._names(record.get("makers") or record.get("maker_names"))
        topics = self._names(record.get("topics") or record.get("tags"))
        votes = self._parse_int(record.get("votes") or record.get("votes_count") or record.get("upvotes"))
        comments_count = self._parse_int(record.get("comments_count") or record.get("comments"))
        saved_at = self._parse_datetime(record.get("saved_at") or record.get("bookmarked_at"))
        featured_at = self._parse_datetime(record.get("featured_at") or record.get("featuredAt"))
        created_at = self._parse_datetime(record.get("created_at") or record.get("createdAt"))
        updated_at = saved_at or featured_at or created_at
        if not name and not url:
            return None
        metadata = {
            "name": name,
            "tagline": tagline,
            "description": description,
            "url": url,
            "makers": makers,
            "topics": topics,
            "votes": votes,
            "comments_count": comments_count,
            "saved_at": saved_at.isoformat() if saved_at else self._text(record.get("saved_at")),
            "featured_at": featured_at.isoformat() if featured_at else self._text(record.get("featured_at")),
            "created_at": created_at.isoformat() if created_at else self._text(record.get("created_at")),
            "source_file": source_file,
            "record": record,
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.PRODUCT_HUNT_BOOKMARKS_JSON,
            source_id=self._source_id(url, name),
            source_entity_type="product_bookmark",
            title=name or url,
            content=self._content(name, tagline, description, url),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(["producthunt", "product_bookmark", *topics])),
            created_at=created_at or featured_at or saved_at or now,
            updated_at=updated_at or now,
        )

    def _content(self, name: str, tagline: str, description: str, url: str) -> str:
        parts = [name, tagline, description, f"URL: {url}" if url else ""]
        return "\n".join(part for part in parts if part)

    def _source_id(self, url: str, name: str) -> str:
        digest = hashlib.sha256((url or name).encode("utf-8")).hexdigest()[:24]
        return f"product_hunt_bookmarks_json:{digest}"

    def _names(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [self._text(item.get("name") if isinstance(item, dict) else item) for item in value if self._text(item.get("name") if isinstance(item, dict) else item)]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return []

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
