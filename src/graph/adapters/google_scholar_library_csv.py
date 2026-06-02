"""Adapter for Google Scholar library CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GoogleScholarLibraryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_scholar_library_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["scholarly_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "scholarly_item" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "title")
        authors = split_values(first(row, "authors", "author"))
        if not any([title, authors, first(row, "url"), first(row, "cluster_id", "cluster id")]):
            return None
        added = parse_datetime(first(row, "added_at", "date added", "added")) or datetime.now(timezone.utc)
        labels = split_values(first(row, "labels", "label"))
        cluster_id = first(row, "cluster_id", "cluster id", "cluster")
        metadata = clean_metadata({"title": title, "authors": authors, "publication": first(row, "publication", "journal", "venue"), "year": parse_int(first(row, "year")), "citations": parse_int(first(row, "citations", "cited by")), "url": first(row, "url", "link"), "cluster_id": cluster_id, "labels": labels, "added_at": added.isoformat(), "source_file": source_file})
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{cluster_id}" if cluster_id else digest_source_id(self.name, title, authors, metadata.get("year"), index), source_entity_type="scholarly_item", title=title or f"Scholar item {index + 1}", content=_content(title, authors, metadata), content_type=ContentType.ARTIFACT, metadata=metadata, tags=[tag for tag in dict.fromkeys(["scholar", "library", *labels]) if tag], created_at=added, updated_at=added)


def _content(title: str, authors: list[str], metadata: dict[str, Any]) -> str:
    return "\n".join(part for part in (title, f"Authors: {', '.join(authors)}" if authors else "", f"Publication: {metadata.get('publication')}" if metadata.get("publication") else "", f"URL: {metadata.get('url')}" if metadata.get("url") else "") if part)
