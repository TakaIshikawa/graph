"""Adapter for local Zotero RDF/XML exports."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
SKIPPED_ITEM_TYPES = {"attachment", "note"}
ITEM_TYPE_NAMES = {
    "article",
    "book",
    "booksection",
    "conferencepaper",
    "document",
    "journalarticle",
    "magazinearticle",
    "manuscript",
    "newspaperarticle",
    "patent",
    "presentation",
    "report",
    "thesis",
    "webpage",
}
TITLE_NAMES = {"title", "shorttitle"}
PUBLICATION_TITLE_NAMES = {
    "publicationtitle",
    "journaltitle",
    "journal",
    "booktitle",
    "container-title",
    "seriestitle",
}
DATE_NAMES = {"date", "issued", "dateaccepted", "datesubmitted"}
DOI_NAMES = {"doi"}
URL_NAMES = {"url", "identifier"}
ABSTRACT_NAMES = {"abstract", "abstractnote", "description"}
CREATOR_CONTAINER_NAMES = {
    "authors",
    "author",
    "creators",
    "creator",
    "contributors",
    "contributor",
    "editors",
    "editor",
}
PERSON_NAMES = {"person", "agent"}
KEY_NAMES = {"itemkey", "key"}
KEYWORD_NAMES = {"subject", "keyword", "tag"}


class ZoteroRdfAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "zotero_rdf"

    @property
    def entity_types(self) -> list[str]:
        return ["zotero_rdf_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "zotero_rdf_item" not in entity_types:
            return result

        paths = self._discover_paths()
        if not paths:
            return result

        sync_at = self._sync_timestamp(since) if since else None
        root = Path(self.path).expanduser()
        root = root if root.is_dir() else root.parent

        for path in paths:
            stat = path.stat()
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            parsed = self._parse(path)
            for item in parsed:
                unit = self._unit_from_item(
                    root,
                    path,
                    item,
                    created_timestamp=stat.st_ctime,
                )
                if unit is not None:
                    result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _discover_paths(self) -> list[Path]:
        configured = Path(self.path).expanduser()
        if configured.is_file() and configured.suffix.lower() in {".rdf", ".xml"}:
            return [configured]
        if configured.is_dir():
            return sorted(
                item
                for item in configured.rglob("*")
                if item.is_file() and item.suffix.lower() in {".rdf", ".xml"}
            )
        return []

    def _parse(self, path: Path) -> list[ET.Element]:
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            raise ValueError(f"Malformed Zotero RDF/XML in {path}: {exc}") from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"Unable to read Zotero RDF/XML from {path}: {exc}") from exc

        return [item for item in tree.getroot() if self._is_supported_item(item)]

    def _is_supported_item(self, item: ET.Element) -> bool:
        item_type = self._item_type(item)
        normalized_type = self._normalize_key(item_type)
        if normalized_type in SKIPPED_ITEM_TYPES:
            return False
        if item_type:
            return True

        tag_name = self._normalize_key(self._local_name(item.tag))
        return tag_name in ITEM_TYPE_NAMES and bool(self._first_text(item, TITLE_NAMES))

    def _unit_from_item(
        self,
        root: Path,
        path: Path,
        item: ET.Element,
        *,
        created_timestamp: float,
    ) -> KnowledgeUnit | None:
        source_file = path.relative_to(root).as_posix()
        title = self._first_text(item, TITLE_NAMES)
        if not title:
            return None

        creators = self._creators(item)
        publication_title = self._first_text(item, PUBLICATION_TITLE_NAMES)
        date_text = self._first_text(item, DATE_NAMES)
        doi = self._doi(item)
        url = self._url(item)
        abstract = self._first_text(item, ABSTRACT_NAMES)
        item_type = self._item_type(item)
        zotero_key = self._zotero_key(item)
        source_id = self._source_id(
            source_file=source_file,
            item=item,
            title=title,
            doi=doi,
            url=url,
            zotero_key=zotero_key,
        )
        parsed_date = self._parse_datetime(date_text)

        metadata = {
            "zotero_key": zotero_key,
            "item_type": item_type,
            "title": title,
            "creators": creators,
            "publication_title": publication_title,
            "date": date_text,
            "doi": doi,
            "url": url,
            "abstract": abstract,
            "source_file": source_file,
        }

        unit = KnowledgeUnit(
            source_project=SourceProject.ZOTERO_RDF,
            source_id=source_id,
            source_entity_type="zotero_rdf_item",
            title=title,
            content=self._content(creators, publication_title, date_text, abstract, doi, url),
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=self._tags(item, item_type),
            created_at=parsed_date or datetime.fromtimestamp(created_timestamp, tz=timezone.utc),
        )
        if parsed_date is not None:
            unit.updated_at = parsed_date
        return unit

    def _content(
        self,
        creators: list[str],
        publication_title: str,
        date_text: str,
        abstract: str,
        doi: str,
        url: str,
    ) -> str:
        parts: list[str] = []
        if creators:
            parts.append(f"Creators: {'; '.join(creators)}")
        if date_text:
            parts.append(f"Date: {date_text}")
        if publication_title:
            parts.append(f"Publication: {publication_title}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        if doi:
            parts.append(f"DOI: {doi}")
        if url:
            parts.append(f"URL: {url}")
        return "\n\n".join(parts)

    def _source_id(
        self,
        *,
        source_file: str,
        item: ET.Element,
        title: str,
        doi: str,
        url: str,
        zotero_key: str,
    ) -> str:
        if zotero_key:
            return f"zotero:{zotero_key}"
        if doi:
            return f"doi:{doi.lower()}"
        if url:
            return f"url:{url}"

        about = self._attribute(item, f"{{{RDF_NS}}}about") or self._attribute(item, "about")
        stable_parts = [
            source_file,
            self._local_name(item.tag),
            about,
            title,
            self._item_type(item),
            self._first_text(item, DATE_NAMES),
        ]
        digest = hashlib.sha256("\n".join(stable_parts).encode("utf-8")).hexdigest()
        return f"zotero_rdf:{digest[:24]}"

    def _item_type(self, item: ET.Element) -> str:
        item_type = self._first_text(item, {"itemtype", "type"})
        if item_type:
            return item_type
        tag_name = self._local_name(item.tag)
        if self._normalize_key(tag_name) in ITEM_TYPE_NAMES:
            return tag_name
        return ""

    def _creators(self, item: ET.Element) -> list[str]:
        creators: list[str] = []
        for container in item:
            if self._normalize_key(self._local_name(container.tag)) not in CREATOR_CONTAINER_NAMES:
                continue
            for person in container.iter():
                if person is container:
                    continue
                local = self._normalize_key(self._local_name(person.tag))
                if local in PERSON_NAMES:
                    name = self._person_name(person)
                    if name and name not in creators:
                        creators.append(name)
                elif local in {"li", "description"} and not list(person):
                    name = self._clean_text(person.text or "")
                    if name and name not in creators:
                        creators.append(name)
        return creators

    def _person_name(self, person: ET.Element) -> str:
        given = self._first_text(person, {"firstname", "givenname", "given"})
        surname = self._first_text(person, {"surname", "familyname", "family"})
        literal = self._first_text(person, {"name"})
        if given and surname:
            return f"{surname}, {given}"
        return surname or given or literal

    def _zotero_key(self, item: ET.Element) -> str:
        key = self._first_text(item, KEY_NAMES)
        if key:
            return key

        about = self._attribute(item, f"{{{RDF_NS}}}about") or self._attribute(item, "about")
        match = re.search(r"(?:/items/|/items/[^/#?]+/)([A-Z0-9]{6,})", about, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""

    def _doi(self, item: ET.Element) -> str:
        doi = self._first_text(item, DOI_NAMES)
        if not doi:
            identifier = self._first_text(item, {"identifier"})
            doi_match = re.search(
                r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\S+)",
                identifier,
                re.IGNORECASE,
            )
            doi = doi_match.group(1) if doi_match else ""
        return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE).strip()

    def _url(self, item: ET.Element) -> str:
        for child in item.iter():
            local = self._normalize_key(self._local_name(child.tag))
            if local not in URL_NAMES:
                continue
            text = self._clean_text(child.text or "")
            if text.startswith(("http://", "https://")) and not re.search(
                r"doi\.org/10\.",
                text,
                re.IGNORECASE,
            ):
                return text
        return ""

    def _tags(self, item: ET.Element, item_type: str) -> list[str]:
        tags: list[str] = []
        for child in item.iter():
            if self._normalize_key(self._local_name(child.tag)) not in KEYWORD_NAMES:
                continue
            for raw_tag in re.split(r"[,;]", self._clean_text(child.text or "")):
                tag = raw_tag.strip().removeprefix("#").strip()
                if tag and tag not in tags and not tag.startswith("10."):
                    tags.append(tag)

        if item_type and item_type not in tags:
            tags.append(item_type)
        return tags

    def _first_text(self, item: ET.Element, names: set[str]) -> str:
        for child in item:
            if self._normalize_key(self._local_name(child.tag)) in names:
                text = self._clean_text(child.text or "")
                if text:
                    return text
        return ""

    def _attribute(self, item: ET.Element, name: str) -> str:
        value = item.attrib.get(name)
        return self._clean_text(value or "")

    def _parse_datetime(self, value: str) -> datetime | None:
        value = value.strip()
        if not value:
            return None

        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            match = re.search(r"\b(\d{4})(?:[-/](\d{1,2}))?(?:[-/](\d{1,2}))?", value)
            if match is None:
                return None
            year = int(match.group(1))
            month = int(match.group(2) or "1")
            day = int(match.group(3) or "1")
            try:
                parsed = datetime(year, month, day)
            except ValueError:
                return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at).replace("Z", "+00:00")).timestamp()

    def _local_name(self, tag: str) -> str:
        if "}" in tag:
            return tag.rsplit("}", 1)[1]
        return tag.rsplit(":", 1)[-1]

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    def _clean_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()
