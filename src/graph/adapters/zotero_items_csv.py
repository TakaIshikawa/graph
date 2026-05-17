"""Adapter for Zotero items CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState

_ITEM_TYPES = {
    "journal article": "article",
    "journalarticle": "article",
    "article": "article",
    "book": "book",
    "book section": "book_section",
    "booksection": "book_section",
    "conference paper": "conference_paper",
    "conferencepaper": "conference_paper",
    "webpage": "webpage",
    "web page": "webpage",
    "report": "report",
    "thesis": "thesis",
}


class ZoteroItemsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "zotero_items_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "item" not in entity_types:
            return result

        sync_at = self._sync_at(since)
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Title", "Name")
        key = first(row, "Key", "Item Key", "ID")
        citation_key = first(row, "Citation Key", "Better BibTeX Citation Key", "Citekey", "CitationKey")
        doi = first(row, "DOI")
        isbn = first(row, "ISBN")
        url = first(row, "URL", "Url")
        abstract = first(row, "Abstract Note", "Abstract", "Summary")
        if not any((title, key, citation_key, doi, isbn, url, abstract)):
            return None

        raw_type = first(row, "Item Type", "Type") or "item"
        item_type = self._item_type(raw_type)
        creators = self._people(first(row, "Creators", "Creator", "Author", "Authors"))
        publication = first(row, "Publication Title", "Journal", "Book Title", "Proceedings Title")
        date = first(row, "Date", "Publication Date")
        tags = self._tags(row)
        collections = split_values(first(row, "Collections", "Collection"))
        added_text = first(row, "Date Added", "Added", "Created")
        modified_text = first(row, "Date Modified", "Modified", "Updated")
        date_at = parse_datetime(date)
        added_at = parse_datetime(added_text)
        modified_at = parse_datetime(modified_text)
        now = datetime.now(timezone.utc)

        return KnowledgeUnit(
            source_project="zotero_items_csv",
            source_id=digest_source_id("zotero_items_csv", key or citation_key or doi.casefold() or isbn or url or title or index),
            source_entity_type="item",
            title=title or citation_key or key or doi or isbn or url or "Untitled Zotero item",
            content=self._content(title, item_type, creators, publication, doi, isbn, url, abstract, date, tags, collections, citation_key),
            content_type=ContentType.ARTIFACT,
            metadata=clean_metadata(
                {
                    "key": key,
                    "citation_key": citation_key,
                    "item_type": item_type,
                    "raw_item_type": raw_type,
                    "title": title,
                    "creators": creators,
                    "authors": creators,
                    "publication_title": publication,
                    "doi": doi,
                    "isbn": isbn,
                    "url": url,
                    "source_url": url,
                    "abstract": abstract,
                    "date": date,
                    "date_added": added_text,
                    "date_modified": modified_text,
                    "added_at": added_at.isoformat() if added_at else "",
                    "modified_at": modified_at.isoformat() if modified_at else "",
                    "tags": tags,
                    "collections": collections,
                    "source_file": source_file,
                    "row": dict(row),
                }
            ),
            tags=tags,
            created_at=added_at or date_at or modified_at or now,
            updated_at=modified_at or added_at or date_at or now,
        )

    def _content(
        self,
        title: str,
        item_type: str,
        creators: list[str],
        publication: str,
        doi: str,
        isbn: str,
        url: str,
        abstract: str,
        date: str,
        tags: list[str],
        collections: list[str],
        citation_key: str,
    ) -> str:
        parts = [title] if title else []
        parts.append(f"Item type: {item_type}")
        if creators:
            parts.append(f"Creators: {'; '.join(creators)}")
        if publication:
            parts.append(f"Publication: {publication}")
        if date:
            parts.append(f"Date: {date}")
        if doi:
            parts.append(f"DOI: {doi}")
        if isbn:
            parts.append(f"ISBN: {isbn}")
        if url:
            parts.append(f"URL: {url}")
        if citation_key:
            parts.append(f"Citation key: {citation_key}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if collections:
            parts.append(f"Collections: {', '.join(collections)}")
        return "\n".join(parts)

    def _item_type(self, value: str) -> str:
        key = value.casefold().replace("_", " ").strip()
        return _ITEM_TYPES.get(key, key.replace(" ", "_") or "item")

    def _people(self, value: str) -> list[str]:
        people: list[str] = []
        for person in split_values(value.replace("\n", ";")):
            normalized = " ".join(person.split())
            if normalized and normalized.casefold() not in {item.casefold() for item in people}:
                people.append(normalized)
        return people

    def _tags(self, row: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        for value in (first(row, "Tags", "Manual Tags"), first(row, "Automatic Tags")):
            for tag in split_values(value):
                normalized = " ".join(tag.casefold().split())
                if normalized and normalized not in tags:
                    tags.append(normalized)
        return tags

    def _sync_at(self, since: SyncState | None) -> datetime | None:
        if since is None:
            return None
        value = since.last_sync_at
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
