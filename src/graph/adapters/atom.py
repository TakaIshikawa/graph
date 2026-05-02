"""Adapter for local Atom XML feed exports."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


ATOM_NS = "{http://www.w3.org/2005/Atom}"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self.parts)


class AtomAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "atom"

    @property
    def entity_types(self) -> list[str]:
        return ["atom_entry"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "atom_entry" not in entity_types:
            return result

        sync_at = self._sync_datetime(since) if since else None
        for path in self._iter_paths():
            try:
                root = ET.parse(path).getroot()
            except (OSError, ET.ParseError, UnicodeDecodeError):
                continue

            for entry in self._entries(root):
                parsed = self._parse_entry(entry, path)
                sync_candidate = parsed["updated_at"] or parsed["published_at"]
                if sync_at and sync_candidate and sync_candidate <= sync_at:
                    continue

                now = datetime.now(timezone.utc)
                created_at = parsed["published_at"] or parsed["updated_at"] or now
                updated_at = parsed["updated_at"] or parsed["published_at"] or now
                result.units.append(
                    KnowledgeUnit(
                        source_project="atom",
                        source_id=parsed["source_id"],
                        source_entity_type="atom_entry",
                        title=parsed["title"],
                        content=parsed["content"],
                        content_type=ContentType.ARTIFACT,
                        metadata=parsed["metadata"],
                        tags=parsed["tags"],
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                )

        return result

    def _iter_paths(self) -> list[Path]:
        path = Path(self.path).expanduser()
        if path.is_dir():
            return sorted(child for child in path.rglob("*.xml") if child.is_file())
        if path.exists() and path.is_file():
            return [path]
        return []

    def _entries(self, root: ET.Element) -> list[ET.Element]:
        if self._local_name(root.tag) == "entry":
            return [root]
        return [
            child
            for child in root
            if self._local_name(child.tag) == "entry"
            and self._namespace(child.tag) in {"", ATOM_NS}
        ]

    def _parse_entry(self, entry: ET.Element, path: Path) -> dict[str, Any]:
        title = self._clean_text(self._child_text(entry, "title")) or "Untitled Atom entry"
        summary = self._child_xml_text(entry, "summary")
        content = self._child_xml_text(entry, "content")
        entry_id = self._child_text(entry, "id")
        links = self._links(entry)
        link_hrefs = [link["href"] for link in links]
        link = self._preferred_link(links)
        published = self._child_text(entry, "published")
        updated = self._child_text(entry, "updated")
        tags = self._categories(entry)
        author = self._author(entry)

        return {
            "source_id": entry_id or link or f"{path}:{title}",
            "title": title,
            "content": self._clean_text(summary or content) or title,
            "published_at": self._parse_datetime(published),
            "updated_at": self._parse_datetime(updated),
            "tags": tags,
            "metadata": {
                "id": entry_id,
                "link": link,
                "link_hrefs": link_hrefs,
                "links": links,
                "author": author,
                "published": published,
                "updated": updated,
                "categories": tags,
                "source_path": str(path),
            },
        }

    def _child_text(self, element: ET.Element, local_name: str) -> str:
        child = self._child(element, local_name)
        if child is None or child.text is None:
            return ""
        return child.text.strip()

    def _child_xml_text(self, element: ET.Element, local_name: str) -> str:
        child = self._child(element, local_name)
        if child is None:
            return ""
        return "".join(child.itertext()).strip()

    def _child(self, element: ET.Element, local_name: str) -> ET.Element | None:
        for child in element:
            if self._local_name(child.tag) == local_name:
                return child
        return None

    def _links(self, entry: ET.Element) -> list[dict[str, str]]:
        links: list[dict[str, str]] = []
        for child in entry:
            if self._local_name(child.tag) != "link":
                continue
            href = child.attrib.get("href", "").strip()
            if not href:
                continue
            links.append(
                {
                    "href": href,
                    "rel": child.attrib.get("rel", "alternate"),
                    "type": child.attrib.get("type", ""),
                    "title": child.attrib.get("title", ""),
                }
            )
        return links

    def _preferred_link(self, links: list[dict[str, str]]) -> str:
        for link in links:
            if link["rel"] == "alternate":
                return link["href"]
        if links:
            return links[0]["href"]
        return ""

    def _author(self, entry: ET.Element) -> str:
        author = self._child(entry, "author")
        if author is None:
            return ""
        name = self._child_text(author, "name")
        email = self._child_text(author, "email")
        uri = self._child_text(author, "uri")
        return name or email or uri

    def _categories(self, entry: ET.Element) -> list[str]:
        tags: list[str] = []
        seen: set[str] = set()
        for child in entry:
            if self._local_name(child.tag) != "category":
                continue
            value = child.attrib.get("term") or child.attrib.get("label") or child.text or ""
            tag = re.sub(r"\s+", " ", value.strip().removeprefix("#")).strip()
            key = tag.casefold()
            if tag and key not in seen:
                tags.append(tag)
                seen.add(key)
        return tags

    def _clean_text(self, value: str) -> str:
        value = unescape(value or "").strip()
        if "<" not in value or ">" not in value:
            return re.sub(r"\s+", " ", value).strip()
        parser = _TextExtractor()
        parser.feed(value)
        text = parser.text() or value
        text = re.sub(r"\s+", " ", unescape(text)).strip()
        return re.sub(r"\s+([,.;:!?])", r"\1", text)

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
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

    def _local_name(self, tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def _namespace(self, tag: str) -> str:
        if tag.startswith("{"):
            return tag.split("}", 1)[0] + "}"
        return ""
