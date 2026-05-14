"""Adapter for Audible library CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import (
    clean_metadata,
    digest_source_id,
    ensure_utc,
    first,
    iter_paths,
    parse_datetime,
    parse_duration_seconds,
    parse_float,
    read_csv_rows,
    split_values,
)
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class AudibleLibraryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "audible_library_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["audiobook"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else set(self.entity_types)
        if "audiobook" not in allowed:
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.updated_at, unit.source_id)))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = first(row, "Title", "Book Title", "Product Name", "Name")
        asin = first(row, "ASIN", "Product ASIN", "Product ID", "ProductId")
        product_url = first(row, "Product URL", "URL", "Link", "Store URL")
        if not (title or asin or product_url):
            return None

        purchase_date = parse_datetime(first(row, "Purchase Date", "Date Purchased", "Purchased", "Date Added"))
        release_date = parse_datetime(first(row, "Release Date", "Publication Date", "Published Date"))
        updated_at = parse_datetime(first(row, "Updated At", "Updated", "Modified At")) or purchase_date or release_date or datetime.now(timezone.utc)
        authors = split_values(first(row, "Author", "Authors", "Written By"))
        narrators = split_values(first(row, "Narrator", "Narrators", "Narrated By"))
        duration = first(row, "Duration", "Length", "Runtime")
        rating = parse_float(first(row, "Rating", "My Rating", "Average Rating"))
        metadata = clean_metadata(
            {
                "title": title,
                "authors": authors,
                "narrators": narrators,
                "purchase_date": purchase_date.isoformat() if purchase_date else None,
                "release_date": release_date.isoformat() if release_date else None,
                "duration": duration,
                "duration_seconds": parse_duration_seconds(duration),
                "rating": rating,
                "asin": asin,
                "product_url": product_url,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.AUDIBLE_LIBRARY_CSV,
            source_id=self._source_id(asin, product_url, title),
            source_entity_type="audiobook",
            title=title or asin or product_url,
            content=self._content(title, authors, narrators, duration, product_url),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["audible", "audiobook"],
            created_at=purchase_date or release_date or updated_at,
            updated_at=updated_at,
        )

    def _source_id(self, asin: str, product_url: str, title: str) -> str:
        if asin:
            return f"audible_library_csv:asin:{asin}"
        return digest_source_id("audible_library_csv", product_url or title)

    def _content(self, title: str, authors: list[str], narrators: list[str], duration: str, product_url: str) -> str:
        parts = [title] if title else []
        if authors:
            parts.append(f"Author: {', '.join(authors)}")
        if narrators:
            parts.append(f"Narrator: {', '.join(narrators)}")
        if duration:
            parts.append(f"Duration: {duration}")
        if product_url:
            parts.append(f"URL: {product_url}")
        return "\n".join(parts)
