"""Adapter for RSS and Atom feeds."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


ATOM_NS = "{http://www.w3.org/2005/Atom}"
CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self.parts)


class FeedAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "feed"

    @property
    def entity_types(self) -> list[str]:
        return ["feed_item"]

    def __init__(self, sources: str = "") -> None:
        self.sources = sources

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "feed_item" not in entity_types:
            return result

        for source in self._iter_sources():
            try:
                root = ET.fromstring(self._read_source(source))
            except (OSError, ET.ParseError):
                continue

            for item in self._parse_feed(root, source):
                created_at = item["created_at"]
                if since and created_at and created_at <= self._sync_datetime(since):
                    continue

                result.units.append(
                    KnowledgeUnit(
                        source_project=SourceProject.ME,
                        source_id=self._source_id(source, item),
                        source_entity_type="feed_item",
                        title=item["title"],
                        content=item["content"],
                        content_type=ContentType.ARTIFACT,
                        metadata={
                            "feed_source": source,
                            "feed_title": item["feed_title"],
                            "id": item["id"],
                            "link": item["link"],
                            "author": item["author"],
                            "published": item["published"],
                            "updated": item["updated"],
                        },
                        tags=item["tags"],
                        created_at=created_at or datetime.now(timezone.utc),
                    )
                )

        return result

    def _iter_sources(self) -> list[str]:
        sources = [
            source.strip()
            for source in re.split(r"[\n,]", self.sources)
            if source.strip()
        ]
        expanded: list[str] = []
        for source in sources:
            if self._is_url(source):
                expanded.append(source)
                continue

            path = Path(source).expanduser()
            if path.is_dir():
                expanded.extend(str(item) for item in sorted(path.rglob("*.xml")))
            elif path.exists():
                expanded.append(str(path))
        return expanded

    def _read_source(self, source: str) -> bytes:
        if self._is_url(source):
            request = Request(source, headers={"User-Agent": "graph-feed-adapter/1.0"})
            with urlopen(request, timeout=15) as response:
                return response.read()
        return Path(source).expanduser().read_bytes()

    def _parse_feed(self, root: ET.Element, source: str) -> list[dict]:
        if root.tag == f"{ATOM_NS}feed" or root.tag.endswith("}feed"):
            return self._parse_atom(root)
        return self._parse_rss(root, source)

    def _parse_rss(self, root: ET.Element, source: str) -> list[dict]:
        channel = root.find("channel")
        if channel is None:
            channel = root
        feed_title = self._child_text(channel, "title") or source
        items: list[dict] = []
        for item in channel.findall("item"):
            title = self._clean_text(self._child_text(item, "title")) or "Untitled feed item"
            description = self._child_text(item, f"{CONTENT_NS}encoded")
            description = description or self._child_text(item, "description")
            published = self._child_text(item, "pubDate")
            updated = self._child_text(item, "updated")
            items.append(
                {
                    "feed_title": feed_title,
                    "title": title,
                    "content": self._clean_text(description) or title,
                    "id": self._guid_text(item),
                    "link": self._child_text(item, "link"),
                    "author": self._child_text(item, "author")
                    or self._child_text_by_local_name(item, "creator"),
                    "tags": self._rss_categories(item),
                    "published": published,
                    "updated": updated,
                    "created_at": self._parse_datetime(published or updated),
                }
            )
        return items

    def _parse_atom(self, root: ET.Element) -> list[dict]:
        feed_title = self._clean_text(self._child_text(root, f"{ATOM_NS}title")) or ""
        items: list[dict] = []
        for entry in root.findall(f"{ATOM_NS}entry"):
            title = self._clean_text(self._child_text(entry, f"{ATOM_NS}title"))
            title = title or "Untitled feed item"
            content = self._child_xml_text(entry, f"{ATOM_NS}content")
            content = content or self._child_xml_text(entry, f"{ATOM_NS}summary")
            published = self._child_text(entry, f"{ATOM_NS}published")
            updated = self._child_text(entry, f"{ATOM_NS}updated")
            items.append(
                {
                    "feed_title": feed_title,
                    "title": title,
                    "content": self._clean_text(content) or title,
                    "id": self._child_text(entry, f"{ATOM_NS}id"),
                    "link": self._atom_link(entry),
                    "author": self._atom_author(entry),
                    "tags": self._atom_categories(entry),
                    "published": published,
                    "updated": updated,
                    "created_at": self._parse_datetime(published or updated),
                }
            )
        return items

    def _child_text(self, element: ET.Element, tag: str) -> str:
        child = element.find(tag)
        if child is None or child.text is None:
            return ""
        return child.text.strip()

    def _child_xml_text(self, element: ET.Element, tag: str) -> str:
        child = element.find(tag)
        if child is None:
            return ""
        return "".join(child.itertext()).strip()

    def _child_text_by_local_name(self, element: ET.Element, local_name: str) -> str:
        for child in element:
            if child.tag.rsplit("}", 1)[-1].split(":", 1)[-1] == local_name and child.text:
                return child.text.strip()
        return ""

    def _guid_text(self, item: ET.Element) -> str:
        guid = item.find("guid")
        if guid is not None and guid.text:
            return guid.text.strip()
        return ""

    def _rss_categories(self, item: ET.Element) -> list[str]:
        tags: list[str] = []
        for category in item.findall("category"):
            if category.text:
                self._append_tag(tags, category.text)
        return tags

    def _atom_categories(self, entry: ET.Element) -> list[str]:
        tags: list[str] = []
        for category in entry.findall(f"{ATOM_NS}category"):
            tag = category.attrib.get("term") or category.attrib.get("label") or ""
            self._append_tag(tags, tag)
        return tags

    def _atom_link(self, entry: ET.Element) -> str:
        fallback = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            href = link.attrib.get("href", "")
            if not href:
                continue
            if link.attrib.get("rel", "alternate") == "alternate":
                return href
            fallback = fallback or href
        return fallback

    def _atom_author(self, entry: ET.Element) -> str:
        author = entry.find(f"{ATOM_NS}author")
        if author is None:
            return ""
        return self._child_text(author, f"{ATOM_NS}name")

    def _append_tag(self, tags: list[str], tag: str) -> None:
        normalized = tag.strip().removeprefix("#").strip()
        if normalized and normalized not in tags:
            tags.append(normalized)

    def _clean_text(self, value: str) -> str:
        value = unescape(value or "").strip()
        if "<" not in value or ">" not in value:
            return re.sub(r"\s+", " ", value).strip()
        parser = _TextExtractor()
        parser.feed(value)
        text = parser.text() or value
        return re.sub(r"\s+", " ", unescape(text)).strip()

    def _parse_datetime(self, value: str) -> datetime | None:
        value = value.strip()
        if not value:
            return None
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _source_id(self, source: str, item: dict) -> str:
        stable_value = item["id"] or item["link"] or item["title"]
        digest = hashlib.sha256(f"{source}\0{stable_value}".encode("utf-8")).hexdigest()
        return f"feed_{digest[:24]}"

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _is_url(self, source: str) -> bool:
        return urlparse(source).scheme in {"http", "https"}
