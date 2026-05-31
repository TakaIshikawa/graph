"""Adapter for Zotero item CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


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
        sync_at = since.last_sync_at if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "title", "item title")
        key = first(row, "key", "zotero key", "item key")
        citation_key = first(row, "citation key", "citekey", "better bibtex citation key")
        item_type = _item_type(first(row, "item type", "type"))
        creators = _split_people(first(row, "creators", "authors", "author"))
        publication = first(row, "publication title", "publication", "journal", "book title")
        date = first(row, "date", "year")
        doi = first(row, "doi")
        isbn = first(row, "isbn")
        url = first(row, "url")
        abstract = first(row, "abstract note", "abstract", "summary")
        tags = _tags(first(row, "tags"))
        collections = split_values(first(row, "collections", "collection"))
        if not any([title, key, doi, isbn, url, abstract]):
            return None
        parsed_date = parse_datetime(date)
        created_at = parse_datetime(first(row, "date added", "created", "created_at")) or parsed_date
        updated_at = parse_datetime(first(row, "date modified", "modified", "updated", "updated_at")) or created_at
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "key": key,
                "zotero_key": key,
                "citation_key": citation_key,
                "item_type": item_type,
                "creators": creators,
                "authors": creators,
                "publication_title": publication,
                "publication": publication,
                "date": date,
                "doi": doi,
                "isbn": isbn,
                "url": url,
                "external_url": url,
                "abstract": abstract,
                "tags": tags,
                "collections": collections,
                "source_file": source_file,
            }
        )
        stable_key = key or citation_key
        return KnowledgeUnit(
            source_project="zotero_items_csv",
            source_id=f"zotero_items_csv:{stable_key}" if stable_key else digest_source_id("zotero_items_csv", doi, isbn, url, title, index),
            source_entity_type="item",
            title=title or doi or url or "Zotero item",
            content=_content(title, creators, publication, date, doi, isbn, url, abstract, tags),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=created_at or now,
            updated_at=updated_at or now,
        )


def _split_people(value: str) -> list[str]:
    return [item for item in split_values(value.replace(" and ", ";"))]


def _tags(value: str) -> list[str]:
    tags: list[str] = []
    for tag in split_values(value):
        normalized = tag.lstrip("#").casefold()
        if normalized and normalized not in tags:
            tags.append(normalized)
    return tags


def _item_type(value: str) -> str:
    text = value.strip()
    if text == "journalArticle":
        return "article"
    return text


def _content(title: str, authors: list[str], publication: str, date: str, doi: str, isbn: str, url: str, abstract: str, tags: list[str]) -> str:
    parts = [title]
    for label, value in (("Creators", "; ".join(authors)), ("Publication", publication), ("Date", date), ("DOI", doi), ("ISBN", isbn), ("URL", url)):
        if value:
            parts.append(f"{label}: {value}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    return "\n".join(part for part in parts if part)
