"""Adapter for Mendeley library CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, parse_int, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class MendeleyLibraryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "mendeley_library_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["document"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "document" not in entity_types:
            return result
        sync_at = since.last_sync_at.astimezone(timezone.utc) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Title", "Document Title", "Name")
        authors = self._people(first(row, "Authors", "Author"))
        year = parse_int(first(row, "Year", "Publication Year"))
        doi = first(row, "DOI", "Doi")
        url = first(row, "URL", "Url", "Link")
        publication = first(row, "Publication", "Published In", "Journal", "Source Title")
        abstract = first(row, "Abstract", "Abstract Note", "Description")
        tags = [tag.casefold() for tag in split_values(first(row, "Tags", "Keywords"))]
        added_at = parse_datetime(first(row, "Date Added", "Added", "Created"))
        modified_at = parse_datetime(first(row, "Date Modified", "Modified", "Updated"))
        if not any((title, authors, doi, url, abstract)):
            return None
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
                "url": url,
                "source_url": url,
                "publication": publication,
                "publication_title": publication,
                "tags": tags,
                "abstract": abstract,
                "date_added": added_at.isoformat() if added_at else "",
                "date_modified": modified_at.isoformat() if modified_at else "",
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="mendeley_library_csv",
            source_id=digest_source_id("mendeley_library_csv", doi.casefold() or url or title or index),
            source_entity_type="document",
            title=title or doi or url or "Untitled Mendeley document",
            content=self._content(title, authors, year, doi, url, publication, tags, abstract),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=added_at or modified_at or now,
            updated_at=modified_at or added_at or now,
        )

    def _people(self, value: str) -> list[str]:
        return [" ".join(person.split()) for person in split_values(value.replace("\n", ";"))]

    def _content(self, title: str, authors: list[str], year: int | None, doi: str, url: str, publication: str, tags: list[str], abstract: str) -> str:
        parts = [title] if title else []
        if authors:
            parts.append(f"Authors: {'; '.join(authors)}")
        if year is not None:
            parts.append(f"Year: {year}")
        if publication:
            parts.append(f"Publication: {publication}")
        if doi:
            parts.append(f"DOI: {doi}")
        if url:
            parts.append(f"URL: {url}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)
