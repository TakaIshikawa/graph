"""Adapter for podcast subscription OPML exports."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from graph.adapters._personal_exports import clean_metadata, iter_paths
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class PodcastsOpmlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "podcasts_opml"

    @property
    def entity_types(self) -> list[str]:
        return ["podcast", "category"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or self.entity_types)
        if not allowed.intersection(self.entity_types):
            return result
        seen: set[str] = set()
        category_podcasts: dict[str, list[KnowledgeUnit]] = {}
        category_names: dict[str, str] = {}
        for path in iter_paths(self.path, {".opml", ".xml"}):
            try:
                root = ET.parse(path).getroot()
            except (OSError, ET.ParseError):
                continue
            body = self._find_child(root, "body")
            if body is None:
                continue
            for unit, categories in self._walk(body, path, ()):
                feed = str(unit.metadata.get("xmlUrl") or "").casefold()
                if feed and feed in seen:
                    continue
                if feed:
                    seen.add(feed)
                for category in categories:
                    category_key = self._category_key(category)
                    if not category_key:
                        continue
                    category_podcasts.setdefault(category_key, []).append(unit)
                    category_names.setdefault(category_key, category)
                if "podcast" in allowed:
                    result.units.append(unit)
        category_units = [
            self._category_unit(category_key, category_names[category_key], category_podcasts[category_key])
            for category_key in sorted(category_podcasts)
        ]
        if "category" in allowed:
            result.units.extend(category_units)
        if {"podcast", "category"}.issubset(allowed):
            category_by_key = {unit.metadata["normalized_name"]: unit for unit in category_units}
            for category_key, podcasts in category_podcasts.items():
                category = category_by_key[category_key]
                for podcast in podcasts:
                    result.edges.append(self._category_edge(podcast, category))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _walk(self, element: ET.Element, source: Path, path_titles: tuple[str, ...]) -> list[tuple[KnowledgeUnit, tuple[str, ...]]]:
        units: list[tuple[KnowledgeUnit, tuple[str, ...]]] = []
        for child in [item for item in element if self._local(item.tag) == "outline"]:
            title = self._title(child)
            xml_url = self._attr(child, "xmlUrl")
            html_url = self._attr(child, "htmlUrl") or self._attr(child, "url")
            description = self._attr(child, "description")
            category = self._attr(child, "category")
            categories = self._category_values(child)
            owner_name = self._attr(child, "ownerName")
            owner_email = self._attr(child, "ownerEmail")
            author = self._attr(child, "author")
            language = self._attr(child, "language")
            image_url = self._attr(child, "imageUrl") or self._attr(child, "imageHref")
            current_path = (*path_titles, title) if title and not xml_url else path_titles
            if xml_url:
                folder_path = tuple(item for item in path_titles if item)
                podcast_categories = tuple(dict.fromkeys([*folder_path, *categories]))
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
                        "categories": list(podcast_categories),
                        "source_file": source.name,
                    }
                )
                tags = list(dict.fromkeys(tag for tag in ["podcast", *podcast_categories, author, language] if tag))
                now = datetime.now(timezone.utc)
                units.append((
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
                    ),
                    podcast_categories,
                ))
            units.extend(self._walk(child, source, current_path))
        return units

    def _source_id(self, xml_url: str) -> str:
        digest = hashlib.sha256(xml_url.strip().casefold().encode("utf-8")).hexdigest()[:24]
        return f"podcasts_opml:{digest}"

    def _category_values(self, outline: ET.Element) -> tuple[str, ...]:
        values: list[str] = []
        for name in ("category", "categories", "genre", "genres"):
            value = self._attr(outline, name)
            if not value:
                continue
            for item in value.replace(";", ",").replace("/", ",").split(","):
                category = " ".join(item.strip().split())
                if category and category.casefold() not in {existing.casefold() for existing in values}:
                    values.append(category)
        return tuple(values)

    def _category_key(self, name: str) -> str:
        normalized = " ".join(name.casefold().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24] if normalized else ""

    def _category_unit(self, category_key: str, name: str, podcasts: list[KnowledgeUnit]) -> KnowledgeUnit:
        podcast_ids = sorted({podcast.source_id for podcast in podcasts})
        source_files = sorted({str(podcast.metadata.get("source_file")) for podcast in podcasts if podcast.metadata.get("source_file")})
        return KnowledgeUnit(
            source_project=SourceProject.PODCASTS_OPML,
            source_id=f"podcasts_opml:category:{category_key}",
            source_entity_type="category",
            title=name,
            content=f"Podcast category: {name}\nPodcasts: {len(podcast_ids)}",
            content_type=ContentType.METADATA,
            metadata={
                "name": name,
                "normalized_name": category_key,
                "podcast_count": len(podcast_ids),
                "podcast_source_ids": podcast_ids,
                "source_files": source_files,
            },
            tags=["podcast-category", name],
            created_at=min(podcast.created_at for podcast in podcasts),
            updated_at=max(podcast.updated_at for podcast in podcasts),
        )

    def _category_edge(self, podcast: KnowledgeUnit, category: KnowledgeUnit) -> KnowledgeEdge:
        digest = hashlib.sha256(f"{podcast.source_id}|{category.source_id}|relates_to".encode("utf-8")).hexdigest()[:24]
        return KnowledgeEdge(
            id=f"podcasts-opml-category-relates-{digest}",
            from_unit_id=podcast.source_id,
            to_unit_id=category.source_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.PODCASTS_OPML.value,
                "from_entity_type": "podcast",
                "to_entity_type": "category",
                "category": category.title,
            },
            created_at=podcast.created_at,
        )

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
