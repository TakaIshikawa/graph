"""Adapter for Zotero library CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
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


class ZoteroLibraryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "zotero_library_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "item" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        if sync_at and sync_at.tzinfo is None:
            sync_at = sync_at.replace(tzinfo=timezone.utc)

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

        result.units.sort(key=lambda unit: (unit.title.casefold(), unit.source_id))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        key = first(row, "Key", "Item Key", "ID")
        title = first(row, "Title", "Name")
        doi = first(row, "DOI")
        url = first(row, "Url", "URL")
        if not any((key, title, doi, url)):
            return None

        raw_type = first(row, "Item Type", "Type") or "item"
        item_type = _ITEM_TYPES.get(raw_type.casefold().replace("_", " "), raw_type.casefold().replace(" ", "_") or "item")
        authors = self._people(first(row, "Author", "Authors", "Creator"))
        year = first(row, "Publication Year", "Year")
        publication = first(row, "Publication Title", "Journal", "Book Title")
        isbn = first(row, "ISBN")
        abstract = first(row, "Abstract Note", "Abstract", "Summary")
        date = first(row, "Date")
        added_text = first(row, "Date Added", "Added", "Created")
        modified_text = first(row, "Date Modified", "Modified", "Updated")
        added_at = parse_datetime(added_text)
        modified_at = parse_datetime(modified_text)
        tags = self._tags(row)
        collections = split_values(first(row, "Collections", "Collection"))
        now = datetime.now(timezone.utc)

        metadata = clean_metadata(
            {
                "key": key,
                "item_type": item_type,
                "raw_item_type": raw_type,
                "title": title or key or doi or url,
                "authors": authors,
                "publication_year": year,
                "publication_title": publication,
                "doi": doi,
                "isbn": isbn,
                "url": url,
                "source_url": url,
                "external_url": url,
                "abstract": abstract,
                "date": date,
                "date_added": added_text,
                "date_modified": modified_text,
                "added_at": added_at.isoformat() if added_at else None,
                "modified_at": modified_at.isoformat() if modified_at else None,
                "tags": tags,
                "collections": collections,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.ZOTERO_LIBRARY_CSV,
            source_id=self._source_id(key, doi, title, url, index),
            source_entity_type="item",
            title=title or key or doi or url or "Untitled Zotero item",
            content=self._content(title, item_type, authors, year, publication, doi, isbn, url, abstract, tags, collections),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=added_at or modified_at or now,
            updated_at=modified_at or added_at or now,
        )

    def _source_id(self, key: str, doi: str, title: str, url: str, index: int) -> str:
        return digest_source_id("zotero_library_csv", key or doi.casefold() or url or title or index)

    def _people(self, value: str) -> list[str]:
        people: list[str] = []
        for person in split_values(value.replace("\n", ";")):
            normalized = " ".join(person.split())
            if normalized and normalized.casefold() not in {item.casefold() for item in people}:
                people.append(normalized)
        return people

    def _tags(self, row: dict[str, Any]) -> list[str]:
        values = [
            first(row, "Tags", "Manual Tags"),
            first(row, "Automatic Tags"),
        ]
        tags: list[str] = []
        for value in values:
            for tag in split_values(value):
                normalized = " ".join(tag.casefold().split())
                if normalized and normalized not in tags:
                    tags.append(normalized)
        return tags

    def _content(
        self,
        title: str,
        item_type: str,
        authors: list[str],
        year: str,
        publication: str,
        doi: str,
        isbn: str,
        url: str,
        abstract: str,
        tags: list[str],
        collections: list[str],
    ) -> str:
        parts = [title] if title else []
        parts.append(f"Item type: {item_type}")
        if authors:
            parts.append(f"Authors: {'; '.join(authors)}")
        if year:
            parts.append(f"Year: {year}")
        if publication:
            parts.append(f"Publication: {publication}")
        if doi:
            parts.append(f"DOI: {doi}")
        if isbn:
            parts.append(f"ISBN: {isbn}")
        if url:
            parts.append(f"URL: {url}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if collections:
            parts.append(f"Collections: {', '.join(collections)}")
        return "\n".join(parts)
