"""Adapter for Mastodon ActivityPub outbox exports."""

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


class MastodonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "mastodon"

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

        path = self._outbox_path()
        if path is None:
            return result

        try:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return result

        sync_at = self._sync_datetime(since) if since else None
        for activity in self._activities(parsed):
            unit = self._unit_from_activity(activity, path.name)
            if unit is None:
                continue
            if sync_at and unit.updated_at <= sync_at:
                continue
            result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _outbox_path(self) -> Path | None:
        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists():
            return None
        if path.is_file():
            return path if path.suffix.lower() == ".json" else None
        if path.is_dir():
            outbox = path / "outbox.json"
            if outbox.is_file():
                return outbox
        return None

    def _activities(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []
        ordered_items = value.get("orderedItems")
        if isinstance(ordered_items, list):
            return [item for item in ordered_items if isinstance(item, dict)]
        items = value.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return [value]

    def _unit_from_activity(
        self,
        activity: dict[str, Any],
        source_file: str,
    ) -> KnowledgeUnit | None:
        if self._first(activity, "type") != "Create":
            return None

        obj = activity.get("object")
        if not isinstance(obj, dict) or self._first(obj, "type") != "Note":
            return None

        content_html = self._first(obj, "content")
        content = self._html_to_text(content_html)
        summary = self._html_to_text(self._first(obj, "summary"))
        title = self._title(content, summary)
        source_id = self._source_id(obj, content)
        published_text = self._first(obj, "published") or self._first(activity, "published")
        updated_text = self._first(obj, "updated") or self._first(activity, "updated")
        published_at = self._parse_datetime(published_text)
        updated_at = self._parse_datetime(updated_text)
        now = datetime.now(timezone.utc)
        tags = self._tags(obj.get("tag"), content)
        to = self._string_list(obj.get("to"))
        cc = self._string_list(obj.get("cc"))

        return KnowledgeUnit(
            source_project=SourceProject.MASTODON,
            source_id=source_id,
            source_entity_type="note",
            title=title,
            content=content or summary or title,
            content_type=ContentType.ARTIFACT,
            metadata={
                "url": self._first(obj, "url"),
                "attributedTo": self._first(obj, "attributedTo"),
                "conversation": self._first(obj, "conversation"),
                "sensitive": self._bool_or_none(obj.get("sensitive")),
                "visibility": self._visibility(to, cc),
                "to": to,
                "cc": cc,
                "published": published_text,
                "updated": updated_text,
                "source_file": source_file,
            },
            tags=tags,
            created_at=published_at or updated_at or now,
            updated_at=updated_at or published_at or now,
        )

    def _source_id(self, obj: dict[str, Any], content: str) -> str:
        source_id = self._first(obj, "id") or self._first(obj, "url")
        if source_id:
            return source_id
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return f"mastodon:{digest[:24]}"

    def _html_to_text(self, value: str) -> str:
        if not value:
            return ""
        parser = _ReadableHTMLParser()
        parser.feed(value)
        parser.close()
        text = unescape(parser.text())
        return re.sub(r"[ \t\r\f\v]+", " ", text).strip()

    def _title(self, content: str, summary: str) -> str:
        title = summary or content
        title = re.sub(r"\s+", " ", title).strip()
        if not title:
            return "Untitled Mastodon note"
        if len(title) <= 80:
            return title
        return f"{title[:77].rstrip()}..."

    def _tags(self, tag_value: Any, content: str) -> list[str]:
        tags: set[str] = set()
        if isinstance(tag_value, list):
            for item in tag_value:
                if isinstance(item, dict) and self._first(item, "type").lower() == "hashtag":
                    tags.add(self._normalize_tag(self._first(item, "name", "href")))
                elif isinstance(item, str):
                    tags.add(self._normalize_tag(item))
        elif isinstance(tag_value, dict):
            tags.add(self._normalize_tag(self._first(tag_value, "name", "href")))

        for match in HASHTAG_RE.finditer(content):
            tags.add(self._normalize_tag(match.group(1)))

        return sorted(tag for tag in tags if tag)

    def _normalize_tag(self, value: str) -> str:
        value = value.strip().removeprefix("#")
        value = re.sub(r"\s+", " ", value).strip().lower()
        return value

    def _visibility(self, to: list[str], cc: list[str]) -> str:
        public = "https://www.w3.org/ns/activitystreams#Public"
        followers = "/followers"
        if public in to:
            return "public"
        if public in cc:
            return "unlisted"
        if any(value.endswith(followers) for value in to + cc):
            return "private"
        return "direct"

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _bool_or_none(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        return None

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)


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
