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
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit:
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "title", "item title")
        key = first(row, "key", "zotero key", "item key")
        item_type = first(row, "item type", "type")
        authors = _split_people(first(row, "creators", "authors", "author"))
        publication = first(row, "publication title", "publication", "journal", "book title")
        date = first(row, "date", "year")
        doi = first(row, "doi")
        isbn = first(row, "isbn")
        url = first(row, "url")
        abstract = first(row, "abstract note", "abstract", "summary")
        tags = split_values(first(row, "tags"))
        collections = split_values(first(row, "collections", "collection"))
        if not any([title, key, doi, isbn, url, abstract]):
            return None
        parsed_date = parse_datetime(date)
        now = datetime.now(timezone.utc)
        metadata = clean_metadata({"zotero_key": key, "item_type": item_type, "authors": authors, "publication": publication, "date": date, "doi": doi, "isbn": isbn, "url": url, "external_url": url, "abstract": abstract, "tags": tags, "collections": collections, "source_file": source_file})
        return KnowledgeUnit(source_project="zotero_items_csv", source_id=f"zotero_items_csv:{key}" if key else digest_source_id("zotero_items_csv", doi, isbn, url, title, index), source_entity_type="item", title=title or doi or url or "Zotero item", content=_content(title, authors, publication, date, doi, isbn, url, abstract, tags), content_type=ContentType.ARTIFACT, metadata=metadata, tags=tags, created_at=parsed_date or now, updated_at=parsed_date or now)


def _split_people(value: str) -> list[str]:
    return [item for item in split_values(value.replace(" and ", ";"))]


def _content(title: str, authors: list[str], publication: str, date: str, doi: str, isbn: str, url: str, abstract: str, tags: list[str]) -> str:
    parts = [title]
    for label, value in (("Authors", ", ".join(authors)), ("Publication", publication), ("Date", date), ("DOI", doi), ("ISBN", isbn), ("URL", url)):
        if value:
            parts.append(f"{label}: {value}")
    if abstract:
        parts.append(abstract)
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    return "\n".join(part for part in parts if part)
