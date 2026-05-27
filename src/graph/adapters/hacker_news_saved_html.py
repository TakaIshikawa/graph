"""Adapter for saved Hacker News HTML pages."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class _HnHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.items: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._capture: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        classes = set(attr.get("class", "").split())
        if tag == "tr" and "athing" in classes:
            self._finish_current()
            self._current = {"hn_item_id": attr.get("id", "")}
        elif self._current is not None and tag == "a":
            href = attr.get("href", "")
            if "titleline" in classes or self._capture == "title":
                self._current["url"] = href
                self._capture = "title"
            elif href.startswith("item?id="):
                self._current["hn_item_url"] = f"https://news.ycombinator.com/{href}"
                self._current["hn_item_id"] = href.split("id=", 1)[1]
                self._capture = "comments"
            elif href.startswith("user?id="):
                self._capture = "author"
        elif self._current is not None and tag == "span":
            if "titleline" in classes:
                self._capture = "title"
            elif "score" in classes:
                self._capture = "points"
            elif "age" in classes:
                self._capture = "age"

    def handle_endtag(self, tag: str) -> None:
        if tag in {"a", "span"}:
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._current is None or self._capture is None:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._capture == "title":
            self._current["title"] = (self._current.get("title", "") + " " + text).strip()
        elif self._capture == "points":
            self._current["points_text"] = text
        elif self._capture == "age":
            self._current["age_text"] = text
        elif self._capture == "author":
            self._current["author"] = text
        elif self._capture == "comments":
            self._current["comments_text"] = text

    def close(self) -> None:
        super().close()
        self._finish_current()

    def _finish_current(self) -> None:
        if self._current and (self._current.get("title") or self._current.get("url") or self._current.get("hn_item_id")):
            self.items.append(self._current)
        self._current = None
        self._capture = None


class HackerNewsSavedHtmlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "hacker_news_saved_html"

    @property
    def entity_types(self) -> list[str]:
        return ["saved_story"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "saved_story" not in set(entity_types or self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None

        for path in self._iter_paths():
            try:
                parser = _HnHtmlParser()
                parser.feed(path.read_text(encoding="utf-8-sig"))
                parser.close()
            except (OSError, UnicodeDecodeError):
                continue
            for item in parser.items:
                unit = self._unit_from_item(item, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file():
            return [root]
        if root.is_dir():
            return sorted(path for suffix in ("*.html", "*.htm") for path in root.rglob(suffix) if path.is_file())
        return []

    def _unit_from_item(self, item: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        hn_item_id = str(item.get("hn_item_id") or "").strip()
        if not title and not url and not hn_item_id:
            return None
        hn_item_url = str(item.get("hn_item_url") or (f"https://news.ycombinator.com/item?id={hn_item_id}" if hn_item_id else ""))
        parsed_date = self._parse_date(str(item.get("age_text") or ""))
        now = datetime.now(timezone.utc)
        comments_count = self._parse_comments(str(item.get("comments_text") or ""))
        metadata = {
            "title": title,
            "url": url,
            "hn_item_id": self._parse_int(hn_item_id),
            "hn_item_url": hn_item_url,
            "points": self._parse_int(str(item.get("points_text") or "")),
            "author": item.get("author"),
            "age_text": item.get("age_text"),
            "comments_count": comments_count,
            "source_file": source_file,
        }
        return KnowledgeUnit(
            source_project=self.name,
            source_id=self._source_id(hn_item_id, url or title),
            source_entity_type="saved_story",
            title=title or url or f"Hacker News item {hn_item_id}",
            content=self._content(title, url, hn_item_url),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None)},
            tags=["hacker_news", "saved_story"],
            created_at=parsed_date or now,
            updated_at=parsed_date or now,
        )

    def _content(self, title: str, url: str, hn_item_url: str) -> str:
        return "\n".join(part for part in (title, f"URL: {url}" if url else "", f"Hacker News: {hn_item_url}" if hn_item_url else "") if part)

    def _source_id(self, hn_item_id: str, fallback: str) -> str:
        if hn_item_id:
            return f"hacker_news_saved_html:{hn_item_id}"
        digest = hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:24]
        return f"hacker_news_saved_html:{digest}"

    def _parse_int(self, value: str) -> int | None:
        match = re.search(r"\d+", value.replace(",", ""))
        return int(match.group(0)) if match else None

    def _parse_comments(self, value: str) -> int:
        if "discuss" in value.casefold():
            return 0
        return self._parse_int(value) or 0

    def _parse_date(self, value: str) -> datetime | None:
        for match in re.finditer(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?", value):
            text = match.group(0)
            for candidate in (text, f"{text}T00:00:00"):
                try:
                    return self._ensure_utc(datetime.fromisoformat(candidate.replace(" ", "T")))
                except ValueError:
                    pass
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
