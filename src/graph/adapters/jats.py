"""Adapter for JATS XML article files."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class JatsAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "jats"

    @property
    def entity_types(self) -> list[str]:
        return ["article"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "article" not in entity_types:
            return result

        sync_at = self._sync_timestamp(since) if since else None
        root_path = Path(self.path).expanduser()
        root = root_path if root_path.is_dir() else root_path.parent

        for path in self._discover_paths():
            try:
                stat = path.stat()
            except OSError:
                continue
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            unit = self._parse_unit(root, path, created_timestamp=stat.st_ctime)
            if unit is not None:
                result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _discover_paths(self) -> list[Path]:
        configured = Path(self.path).expanduser()
        if configured.is_file() and configured.suffix.lower() == ".xml":
            return [configured]
        if configured.is_dir():
            return sorted(
                item
                for item in configured.rglob("*")
                if item.is_file() and item.suffix.lower() == ".xml"
            )
        return []

    def _parse_unit(
        self,
        root: Path,
        path: Path,
        *,
        created_timestamp: float,
    ) -> KnowledgeUnit | None:
        try:
            article = ET.parse(path).getroot()
        except (ET.ParseError, OSError, UnicodeDecodeError):
            return None

        article_meta = self._article_meta(article)
        title = self._article_title(article_meta) if article_meta is not None else ""
        if article_meta is None or not title:
            return None

        source_path = path.relative_to(root).as_posix()
        doi = self._doi(article_meta)
        authors = self._authors(article_meta)
        date_parts = self._published_date_parts(article_meta)
        published_at = self._date_from_parts(date_parts)
        journal_title = self._journal_title(article)
        publisher_name = self._publisher_name(article)
        article_type = self._attribute(article, "article-type")

        metadata = {
            "doi": doi,
            "journal_title": journal_title,
            "publisher_name": publisher_name,
            "authors": authors,
            "published_date_parts": date_parts,
            "path": source_path,
            "article_type": article_type,
        }

        unit = KnowledgeUnit(
            source_project=SourceProject.JATS,
            source_id=self._source_id(source_path, doi),
            source_entity_type="article",
            title=title,
            content=self._content(article, authors, journal_title, doi),
            content_type=ContentType.FINDING,
            metadata=metadata,
            tags=self._tags(article_meta),
            created_at=published_at or datetime.fromtimestamp(created_timestamp, tz=timezone.utc),
        )
        if published_at is not None:
            unit.updated_at = published_at
        return unit

    def _article_meta(self, root: ET.Element) -> ET.Element | None:
        if self._local_name(root.tag) != "article":
            return None
        front = self._first_child(root, "front")
        if front is None:
            return None
        return self._first_child(front, "article-meta")

    def _article_title(self, article_meta: ET.Element) -> str:
        title_group = self._first_child(article_meta, "title-group")
        if title_group is None:
            return ""
        article_title = self._first_child(title_group, "article-title")
        return self._element_text(article_title) if article_title is not None else ""

    def _doi(self, article_meta: ET.Element) -> str:
        for article_id in self._children(article_meta, "article-id"):
            if self._attribute(article_id, "pub-id-type").lower() == "doi":
                return self._clean_doi(self._element_text(article_id))
        return ""

    def _journal_title(self, article: ET.Element) -> str:
        journal_meta = self._first_descendant(article, "journal-meta")
        if journal_meta is None:
            return ""
        for name in ("journal-title", "journal-id"):
            value = self._first_descendant_text(journal_meta, name)
            if value:
                return value
        return ""

    def _publisher_name(self, article: ET.Element) -> str:
        journal_meta = self._first_descendant(article, "journal-meta")
        if journal_meta is None:
            return ""
        return self._first_descendant_text(journal_meta, "publisher-name")

    def _authors(self, article_meta: ET.Element) -> list[str]:
        authors: list[str] = []
        for contrib in self._descendants(article_meta, "contrib"):
            if self._attribute(contrib, "contrib-type").lower() not in {"", "author"}:
                continue
            name = self._person_name(contrib)
            if name and name not in authors:
                authors.append(name)
        return authors

    def _person_name(self, contrib: ET.Element) -> str:
        collab = self._first_descendant_text(contrib, "collab")
        if collab:
            return collab

        name = self._first_descendant(contrib, "name")
        if name is not None:
            surname = self._first_child_text(name, "surname")
            given = self._first_child_text(name, "given-names")
            if given and surname:
                return f"{given} {surname}"
            return surname or given or self._element_text(name)

        string_name = self._first_descendant_text(contrib, "string-name")
        return string_name

    def _published_date_parts(self, article_meta: ET.Element) -> dict[str, int]:
        for pub_date in self._children(article_meta, "pub-date"):
            parts = {
                name: value
                for name in ("year", "month", "day")
                if (value := self._int_text(self._first_child_text(pub_date, name))) is not None
            }
            if parts:
                return parts
        return {}

    def _date_from_parts(self, parts: dict[str, int]) -> datetime | None:
        year = parts.get("year")
        if year is None:
            return None
        month = parts.get("month", 1)
        day = parts.get("day", 1)
        try:
            return datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            return None

    def _content(
        self,
        article: ET.Element,
        authors: list[str],
        journal_title: str,
        doi: str,
    ) -> str:
        parts: list[str] = []
        if authors:
            parts.append(f"Authors: {'; '.join(authors)}")
        if journal_title:
            parts.append(f"Journal: {journal_title}")

        abstracts = self._section_paragraphs(article, "abstract")
        if abstracts:
            parts.append("Abstract:\n" + "\n\n".join(abstracts))

        body = self._section_paragraphs(article, "body")
        if body:
            parts.append("Body:\n" + "\n\n".join(body))

        if doi:
            parts.append(f"DOI: {doi}")
        return "\n\n".join(parts)

    def _section_paragraphs(self, article: ET.Element, section_name: str) -> list[str]:
        section = self._first_descendant(article, section_name)
        if section is None:
            return []

        paragraphs = [
            text
            for paragraph in self._descendants(section, "p")
            if (text := self._element_text(paragraph))
        ]
        if paragraphs:
            return paragraphs

        text = self._element_text(section)
        return [text] if text else []

    def _tags(self, article_meta: ET.Element) -> list[str]:
        categories = self._first_child(article_meta, "article-categories")
        if categories is None:
            return []

        tags: list[str] = []
        for subject in self._descendants(categories, "subject"):
            tag = self._element_text(subject)
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _source_id(self, source_path: str, doi: str) -> str:
        if doi:
            return f"doi:{doi.lower()}"
        digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()
        return f"jats:{digest[:24]}"

    def _children(self, element: ET.Element, local_name: str) -> list[ET.Element]:
        return [child for child in list(element) if self._local_name(child.tag) == local_name]

    def _descendants(self, element: ET.Element, local_name: str) -> list[ET.Element]:
        return [item for item in element.iter() if self._local_name(item.tag) == local_name]

    def _first_child(self, element: ET.Element, local_name: str) -> ET.Element | None:
        for child in element:
            if self._local_name(child.tag) == local_name:
                return child
        return None

    def _first_descendant(self, element: ET.Element, local_name: str) -> ET.Element | None:
        for item in element.iter():
            if self._local_name(item.tag) == local_name:
                return item
        return None

    def _first_child_text(self, element: ET.Element, local_name: str) -> str:
        child = self._first_child(element, local_name)
        return self._element_text(child) if child is not None else ""

    def _first_descendant_text(self, element: ET.Element, local_name: str) -> str:
        descendant = self._first_descendant(element, local_name)
        return self._element_text(descendant) if descendant is not None else ""

    def _attribute(self, element: ET.Element, name: str) -> str:
        for key, value in element.attrib.items():
            if self._local_name(key) == name:
                return value.strip()
        return ""

    def _element_text(self, element: ET.Element | None) -> str:
        if element is None:
            return ""
        return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()

    def _local_name(self, tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def _clean_doi(self, value: str) -> str:
        return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE).strip()

    def _int_text(self, value: str) -> int | None:
        try:
            return int(value)
        except ValueError:
            return None

    def _sync_timestamp(self, since: SyncState) -> float:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).timestamp()
