"""Adapter for OPML outline exports."""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


@dataclass
class _OutlineUnit:
    source_id: str
    title: str
    content: str
    metadata: dict
    tags: list[str]


class OpmlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "opml"

    @property
    def entity_types(self) -> list[str]:
        return ["outline"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "outline" not in entity_types:
            return result

        sources = self._iter_sources()
        if not sources:
            warnings.warn("No OPML files found for configured GRAPH_OPML_PATH.", stacklevel=2)
            return result

        for source in sources:
            try:
                root = ET.parse(source).getroot()
            except (OSError, ET.ParseError) as exc:
                warnings.warn(f"Skipping invalid OPML file {source}: {exc}", stacklevel=2)
                continue

            body = self._find_child(root, "body")
            if body is None:
                warnings.warn(f"Skipping OPML file without a body: {source}", stacklevel=2)
                continue

            for index, outline in enumerate(self._outline_children(body), 1):
                self._ingest_outline(
                    outline,
                    source=source,
                    position=(index,),
                    path_titles=(),
                    parent_source_id=None,
                    result=result,
                )

        return result

    def _iter_sources(self) -> list[Path]:
        if not self.path:
            return []

        sources: list[Path] = []
        for raw in re.split(r"[\n,]", self.path):
            if not raw.strip():
                continue
            path = Path(raw.strip()).expanduser()
            if path.is_dir():
                sources.extend(sorted(path.rglob("*.opml")))
                sources.extend(sorted(path.rglob("*.xml")))
            elif path.exists() and path.is_file():
                sources.append(path)

        deduped: list[Path] = []
        seen: set[Path] = set()
        for source in sources:
            resolved = source.resolve()
            if resolved not in seen:
                seen.add(resolved)
                deduped.append(source)
        return deduped

    def _ingest_outline(
        self,
        outline: ET.Element,
        *,
        source: Path,
        position: tuple[int, ...],
        path_titles: tuple[str, ...],
        parent_source_id: str | None,
        result: IngestResult,
    ) -> None:
        unit = self._outline_unit(outline, source=source, position=position, path_titles=path_titles)
        current_source_id = unit.source_id if unit else None

        if unit is not None:
            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.OPML,
                    source_id=unit.source_id,
                    source_entity_type="outline",
                    title=unit.title,
                    content=unit.content,
                    content_type=ContentType.ARTIFACT,
                    metadata=unit.metadata,
                    tags=unit.tags,
                    created_at=datetime.now(timezone.utc),
                )
            )
            if parent_source_id:
                result.edges.append(
                    KnowledgeEdge(
                        from_unit_id=parent_source_id,
                        to_unit_id=unit.source_id,
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.OPML.value,
                            "from_entity_type": "outline",
                            "to_entity_type": "outline",
                            "opml_path": unit.metadata["path"],
                        },
                    )
                )

        next_path_titles = path_titles
        title = self._title(outline)
        if title:
            next_path_titles = (*path_titles, title)

        for index, child in enumerate(self._outline_children(outline), 1):
            self._ingest_outline(
                child,
                source=source,
                position=(*position, index),
                path_titles=next_path_titles,
                parent_source_id=current_source_id or parent_source_id,
                result=result,
            )

    def _outline_unit(
        self,
        outline: ET.Element,
        *,
        source: Path,
        position: tuple[int, ...],
        path_titles: tuple[str, ...],
    ) -> _OutlineUnit | None:
        title = self._title(outline)
        url = self._attr(outline, "url")
        xml_url = self._attr(outline, "xmlUrl")
        html_url = self._attr(outline, "htmlUrl")
        if not any([title, url, xml_url, html_url]):
            return None

        path = (*path_titles, title or url or xml_url or html_url)
        source_id = self._source_id(source, position, title, url, xml_url, html_url)
        metadata = {
            "source_file": str(source),
            "position": ".".join(str(part) for part in position),
            "path": "/".join(path),
            "path_parts": list(path),
            "url": url,
            "xmlUrl": xml_url,
            "htmlUrl": html_url,
            "type": self._attr(outline, "type"),
        }
        metadata = {key: value for key, value in metadata.items() if value not in ("", [])}

        content_parts = [title]
        if path:
            content_parts.append("/".join(path))
        content_parts.extend([url, xml_url, html_url])

        return _OutlineUnit(
            source_id=source_id,
            title=title or url or xml_url or html_url or "Untitled outline",
            content="\n".join(part for part in content_parts if part),
            metadata=metadata,
            tags=self._path_tags(path),
        )

    def _source_id(
        self,
        source: Path,
        position: tuple[int, ...],
        title: str,
        url: str,
        xml_url: str,
        html_url: str,
    ) -> str:
        raw = "|".join(
            [
                str(source.resolve()),
                ".".join(str(part) for part in position),
                title,
                url,
                xml_url,
                html_url,
            ]
        )
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"outline-{'.'.join(str(part) for part in position)}-{digest}"

    def _path_tags(self, path: tuple[str, ...]) -> list[str]:
        tags: list[str] = []
        for index in range(len(path)):
            tag = "/".join(part for part in path[: index + 1] if part)
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _title(self, outline: ET.Element) -> str:
        return self._attr(outline, "text") or self._attr(outline, "title")

    def _attr(self, outline: ET.Element, name: str) -> str:
        return (outline.attrib.get(name) or "").strip()

    def _outline_children(self, element: ET.Element) -> list[ET.Element]:
        return [child for child in element if self._local_name(child.tag) == "outline"]

    def _find_child(self, element: ET.Element, name: str) -> ET.Element | None:
        for child in element:
            if self._local_name(child.tag) == name:
                return child
        return None

    def _local_name(self, tag: str) -> str:
        return tag.rsplit("}", 1)[-1].split(":", 1)[-1]
