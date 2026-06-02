"""Adapter for RSS subscription OPML exports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from graph.adapters._personal_exports import clean_metadata, digest_source_id, iter_paths
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class RssSubscriptionsOpmlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "rss_subscriptions_opml"

    @property
    def entity_types(self) -> list[str]:
        return ["feed"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "feed" not in entity_types:
            return result
        for path in iter_paths(self.path, {".opml", ".xml"}):
            try:
                root = ET.parse(path).getroot()
            except (OSError, ET.ParseError):
                continue
            body = _child(root, "body")
            if body is None:
                continue
            for index, outline in enumerate(list(body)):
                self._walk(outline, path, (index,), (), result)
        return result

    def _walk(self, outline: ET.Element, source: Path, position: tuple[int, ...], categories: tuple[str, ...], result: IngestResult) -> None:
        title = _attr(outline, "title") or _attr(outline, "text")
        xml_url = _attr(outline, "xmlUrl")
        html_url = _attr(outline, "htmlUrl")
        outline_type = _attr(outline, "type")
        category_attr = _attr(outline, "category")
        current_categories = (*categories, title) if title and not xml_url and list(outline) else categories
        category_path = (*categories, *(part.strip() for part in category_attr.split("/") if part.strip()))
        if xml_url:
            now = datetime.now(timezone.utc)
            metadata = clean_metadata(
                {
                    "xml_url": xml_url,
                    "html_url": html_url,
                    "category_path": list(category_path),
                    "outline_type": outline_type,
                    "source_file": source.name,
                }
            )
            result.units.append(
                KnowledgeUnit(
                    source_project="rss_subscriptions_opml",
                    source_id=digest_source_id("rss_subscriptions_opml", xml_url, html_url, title),
                    source_entity_type="feed",
                    title=title or xml_url,
                    content=_content(title, xml_url, html_url),
                    content_type=ContentType.ARTIFACT,
                    metadata=metadata,
                    tags=list(category_path),
                    created_at=now,
                    updated_at=now,
                )
            )
        for child_index, child in enumerate(list(outline)):
            self._walk(child, source, (*position, child_index), current_categories, result)


def _child(root: ET.Element, name: str) -> ET.Element | None:
    for child in root:
        if child.tag.rsplit("}", 1)[-1].casefold() == name:
            return child
    return None


def _attr(outline: ET.Element, name: str) -> str:
    for key, value in outline.attrib.items():
        if key.casefold() == name.casefold():
            return value.strip()
    return ""


def _content(title: str, xml_url: str, html_url: str) -> str:
    parts = [title, f"Feed: {xml_url}" if xml_url else "", f"Site: {html_url}" if html_url else ""]
    return "\n".join(part for part in parts if part)
