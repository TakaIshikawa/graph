"""Adapter for podcast subscription OPML exports."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from graph.adapters._personal_exports import clean_metadata, iter_paths
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class PodcastsOpmlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "podcasts_opml"

    @property
    def entity_types(self) -> list[str]:
        return ["podcast"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types and "podcast" not in entity_types:
            return result
        seen: set[str] = set()
        for path in iter_paths(self.path, {".opml", ".xml"}):
            try:
                root = ET.parse(path).getroot()
            except (OSError, ET.ParseError):
                continue
            body = self._find_child(root, "body")
            if body is None:
                continue
            for unit in self._walk(body, path, ()):
                feed = str(unit.metadata.get("xmlUrl") or "").casefold()
                if feed and feed in seen:
                    continue
                if feed:
                    seen.add(feed)
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _walk(self, element: ET.Element, source: Path, path_titles: tuple[str, ...]) -> list[KnowledgeUnit]:
        units: list[KnowledgeUnit] = []
        for child in [item for item in element if self._local(item.tag) == "outline"]:
            title = self._title(child)
            xml_url = self._attr(child, "xmlUrl")
            html_url = self._attr(child, "htmlUrl") or self._attr(child, "url")
            description = self._attr(child, "description")
            category = self._attr(child, "category")
            owner_name = self._attr(child, "ownerName")
            owner_email = self._attr(child, "ownerEmail")
            author = self._attr(child, "author")
            language = self._attr(child, "language")
            image_url = self._attr(child, "imageUrl") or self._attr(child, "imageHref")
            current_path = (*path_titles, title) if title and not xml_url else path_titles
            if xml_url:
                folder_path = tuple(item for item in path_titles if item)
                metadata = clean_metadata(
                    {
                        "title": title,
                        "xmlUrl": xml_url,
                        "htmlUrl": html_url,
                        "description": description,
                        "category": category,
                        "ownerName": owner_name,
                        "ownerEmail": owner_email,
                        "author": author,
                        "language": language,
                        "imageUrl": image_url,
                        "folder_path": list(folder_path),
                        "source_file": source.name,
                    }
                )
                tags = list(dict.fromkeys(tag for tag in ["podcast", *folder_path, category, author, language] if tag))
                now = datetime.now(timezone.utc)
                units.append(
                    KnowledgeUnit(
                        source_project=SourceProject.PODCASTS_OPML,
                        source_id=self._source_id(xml_url),
                        source_entity_type="podcast",
                        title=title or xml_url,
                        content="\n".join(part for part in [title, description, xml_url, html_url] if part),
                        content_type=ContentType.ARTIFACT,
                        metadata=metadata,
                        tags=tags,
                        created_at=now,
                        updated_at=now,
                    )
                )
            units.extend(self._walk(child, source, current_path))
        return units

    def _source_id(self, xml_url: str) -> str:
        digest = hashlib.sha256(xml_url.strip().casefold().encode("utf-8")).hexdigest()[:24]
        return f"podcasts_opml:{digest}"

    def _title(self, outline: ET.Element) -> str:
        return self._attr(outline, "text") or self._attr(outline, "title")

    def _attr(self, outline: ET.Element, name: str) -> str:
        return (outline.attrib.get(name) or "").strip()

    def _find_child(self, element: ET.Element, name: str) -> ET.Element | None:
        for child in element:
            if self._local(child.tag) == name:
                return child
        return None

    def _local(self, tag: str) -> str:
        return tag.rsplit("}", 1)[-1].split(":", 1)[-1]
