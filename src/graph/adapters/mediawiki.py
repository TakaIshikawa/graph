"""Adapter for local MediaWiki XML dump exports."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


CATEGORY_LINK_RE = re.compile(r"\[\[\s*Category\s*:\s*([^\]|#]+)(?:[^\]]*)\]\]", re.IGNORECASE)


class MediaWikiAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "mediawiki"

    @property
    def entity_types(self) -> list[str]:
        return ["mediawiki_page"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "mediawiki_page" not in entity_types:
            return result

        sync_at = self._sync_datetime(since) if since else None
        for path in self._iter_paths():
            try:
                units = self._read_units(path)
            except (OSError, ET.ParseError):
                continue
            for unit in units:
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        path = Path(self.path).expanduser()
        if path.is_dir():
            return sorted(item for item in path.rglob("*.xml") if item.is_file())
        if path.exists() and path.is_file():
            return [path]
        return []

    def _read_units(self, path: Path) -> list[KnowledgeUnit]:
        units: list[KnowledgeUnit] = []
        for _, page in ET.iterparse(path, events=("end",)):
            if self._local_name(page.tag) != "page":
                continue
            unit = self._unit_from_page(page, path.name)
            if unit is not None:
                units.append(unit)
            page.clear()
        return units

    def _unit_from_page(self, page: ET.Element, source_file: str) -> KnowledgeUnit | None:
        title = self._child_text(page, "title")
        page_id = self._child_text(page, "id")
        namespace = self._child_text(page, "ns")
        redirect = self._child(page, "redirect")
        redirect_target = redirect.attrib.get("title", "").strip() if redirect is not None else ""
        revision = self._current_revision(page)
        if revision is None:
            return None

        text_element = self._child(revision, "text")
        if text_element is None or "deleted" in text_element.attrib:
            return None

        text = text_element.text or ""
        if not text.strip():
            return None

        revision_id = self._child_text(revision, "id")
        timestamp_text = self._child_text(revision, "timestamp")
        timestamp = self._parse_datetime(timestamp_text)
        contributor, contributor_id = self._contributor(revision)
        tags = self._category_tags(text)
        source_id = self._source_id(page_id, revision_id, title)

        metadata = {
            "page_id": page_id,
            "namespace": namespace,
            "revision_id": revision_id,
            "contributor": contributor,
            "timestamp": timestamp_text,
            "redirect_target": redirect_target,
            "source_title": title,
            "source_file": source_file,
        }
        if contributor_id:
            metadata["contributor_id"] = contributor_id

        return KnowledgeUnit(
            source_project=SourceProject.MEDIAWIKI,
            source_id=source_id,
            source_entity_type="mediawiki_page",
            title=title or source_id,
            content=text,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _current_revision(self, page: ET.Element) -> ET.Element | None:
        revision = None
        for child in page:
            if self._local_name(child.tag) == "revision":
                revision = child
        return revision

    def _contributor(self, revision: ET.Element) -> tuple[str, str]:
        contributor = self._child(revision, "contributor")
        if contributor is None:
            return "", ""
        name = self._child_text(contributor, "username") or self._child_text(contributor, "ip")
        return name, self._child_text(contributor, "id")

    def _category_tags(self, text: str) -> list[str]:
        tags: list[str] = []
        seen: set[str] = set()
        for match in CATEGORY_LINK_RE.finditer(text):
            tag = self._normalize_category(match.group(1))
            key = tag.casefold()
            if tag and key not in seen:
                tags.append(tag)
                seen.add(key)
        return tags

    def _normalize_category(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.replace("_", " ")).strip()

    def _source_id(self, page_id: str, revision_id: str, title: str) -> str:
        if page_id and revision_id:
            return f"mediawiki:{page_id}:{revision_id}"
        if page_id:
            return f"mediawiki:{page_id}"
        return f"mediawiki:{title}"

    def _child(self, element: ET.Element, name: str) -> ET.Element | None:
        for child in element:
            if self._local_name(child.tag) == name:
                return child
        return None

    def _child_text(self, element: ET.Element, name: str) -> str:
        child = self._child(element, name)
        return (child.text or "").strip() if child is not None else ""

    def _local_name(self, tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def _parse_datetime(self, value: str) -> datetime:
        if value:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
            else:
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
        return datetime.now(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
