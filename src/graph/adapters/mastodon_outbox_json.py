"""Adapter for Mastodon ActivityPub outbox JSON exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


HASHTAG_RE = re.compile(r"(?<![\w/])#([\w][\w-]*)", re.UNICODE)
PUBLIC_AUDIENCE = "https://www.w3.org/ns/activitystreams#Public"


class MastodonOutboxJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "mastodon_outbox_json"

    @property
    def entity_types(self) -> list[str]:
        return ["note"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "note" not in entity_types:
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                parsed = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, activity in enumerate(self._activities(parsed)):
                unit = self._unit_from_activity(activity, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.created_at, unit.source_id)))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        outbox = root / "outbox.json"
        if outbox.is_file():
            return [outbox]
        return sorted(path for path in root.rglob("*.json") if path.is_file())

    def _activities(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []
        items = value.get("orderedItems")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        items = value.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return [value]

    def _unit_from_activity(
        self,
        activity: dict[str, Any],
        source_file: str,
        record_index: int,
    ) -> KnowledgeUnit | None:
        if self._text(activity.get("type")) != "Create":
            return None
        note = activity.get("object")
        if not isinstance(note, dict) or self._text(note.get("type")) != "Note":
            return None

        content = self._html_to_text(self._text(note.get("content")))
        summary = self._html_to_text(self._text(note.get("summary")))
        published_text = self._text(note.get("published") or activity.get("published"))
        updated_text = self._text(note.get("updated") or activity.get("updated"))
        published_at = self._parse_datetime(published_text)
        updated_at = self._parse_datetime(updated_text)
        now = datetime.now(timezone.utc)
        note_id = self._text(note.get("id"))
        activity_id = self._text(activity.get("id"))
        url = self._url(note.get("url"))
        to = self._string_list(note.get("to"))
        cc = self._string_list(note.get("cc"))
        tags = self._tags(note.get("tag"), content)

        metadata = {
            "activity_id": activity_id,
            "activity_type": self._text(activity.get("type")),
            "note_id": note_id,
            "url": url,
            "published": published_text,
            "updated": updated_text,
            "visibility": self._visibility(to, cc),
            "to": to,
            "cc": cc,
            "audience": self._string_list(note.get("audience")),
            "tag": self._tag_metadata(note.get("tag")),
            "tags": tags,
            "in_reply_to": self._text(note.get("inReplyTo")),
            "attributed_to": self._text(note.get("attributedTo")),
            "conversation": self._text(note.get("conversation")),
            "sensitive": note.get("sensitive") if isinstance(note.get("sensitive"), bool) else None,
            "replies": note.get("replies") if isinstance(note.get("replies"), dict) else None,
            "likes": note.get("likes") if isinstance(note.get("likes"), dict) else None,
            "shares": note.get("shares") if isinstance(note.get("shares"), dict) else None,
            "source_file": source_file,
            "record_index": record_index,
        }

        return KnowledgeUnit(
            source_project=SourceProject.MASTODON_OUTBOX_JSON,
            source_id=self._source_id(activity_id, note_id, url, content, record_index),
            source_entity_type="note",
            title=self._title(content, summary),
            content=content or summary or "Untitled Mastodon note",
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=tags,
            created_at=published_at or updated_at or now,
            updated_at=updated_at or published_at or now,
        )

    def _source_id(self, activity_id: str, note_id: str, url: str, content: str, index: int) -> str:
        if note_id:
            return note_id
        if activity_id:
            return activity_id
        if url:
            return url
        digest = hashlib.sha256(f"{content}|{index}".encode("utf-8")).hexdigest()[:24]
        return f"mastodon_outbox_json:{digest}"

    def _html_to_text(self, value: str) -> str:
        if not value:
            return ""
        parser = _ReadableHTMLParser()
        parser.feed(value)
        parser.close()
        text = unescape(parser.text())
        return re.sub(r"[ \t\r\f\v]+", " ", text).strip()

    def _title(self, content: str, summary: str) -> str:
        title = re.sub(r"\s+", " ", summary or content).strip()
        if not title:
            return "Untitled Mastodon note"
        return title if len(title) <= 80 else f"{title[:77].rstrip()}..."

    def _tags(self, tag_value: Any, content: str) -> list[str]:
        tags: set[str] = set()
        for item in self._as_list(tag_value):
            if isinstance(item, dict) and self._text(item.get("type")).lower() == "hashtag":
                tags.add(self._normalize_tag(item.get("name") or item.get("href")))
            elif isinstance(item, str):
                tags.add(self._normalize_tag(item))
        for match in HASHTAG_RE.finditer(content):
            tags.add(self._normalize_tag(match.group(1)))
        return sorted(tag for tag in tags if tag)

    def _tag_metadata(self, tag_value: Any) -> list[dict[str, Any]]:
        tags: list[dict[str, Any]] = []
        for item in self._as_list(tag_value):
            if isinstance(item, dict):
                tags.append({str(key): value for key, value in item.items() if value not in ("", None, [])})
            elif isinstance(item, str) and item.strip():
                tags.append({"name": item.strip()})
        return tags

    def _visibility(self, to: list[str], cc: list[str]) -> str:
        if PUBLIC_AUDIENCE in to:
            return "public"
        if PUBLIC_AUDIENCE in cc:
            return "unlisted"
        if any(value.endswith("/followers") for value in to + cc):
            return "private"
        return "direct"

    def _url(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            return self._text(value.get("href") or value.get("id"))
        for item in self._as_list(value):
            if isinstance(item, dict):
                url = self._text(item.get("href") or item.get("id"))
                if url:
                    return url
            if isinstance(item, str) and item.strip():
                return item.strip()
        return ""

    def _string_list(self, value: Any) -> list[str]:
        return [str(item).strip() for item in self._as_list(value) if str(item).strip()]

    def _as_list(self, value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if value in (None, ""):
            return []
        return [value]

    def _normalize_tag(self, value: Any) -> str:
        return re.sub(r"\s+", " ", self._text(value).removeprefix("#")).strip().lower()

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _text(self, value: Any) -> str:
        if isinstance(value, (dict, list)) or value is None:
            return ""
        return str(value).strip()


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "p", "div", "li", "blockquote"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "li", "blockquote"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def text(self) -> str:
        lines = [re.sub(r"\s+", " ", part).strip() for part in "".join(self._parts).splitlines()]
        return "\n".join(line for line in lines if line)
