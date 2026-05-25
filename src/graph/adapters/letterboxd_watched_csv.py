"""Adapter for Letterboxd watched/ratings CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, parse_float, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class LetterboxdWatchedCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "letterboxd_watched_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["film"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "film" not in entity_types:
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

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Name", "Title", "Film")
        year = first(row, "Year", "Release Year")
        uri = first(row, "Letterboxd URI", "Letterboxd URI ", "URI", "URL")
        if not any((title, uri)):
            return None

        watched_text = first(row, "Watched Date", "Date", "Watched")
        watched_at = parse_datetime(watched_text)
        rating = parse_float(first(row, "Rating", "Stars"))
        review = first(row, "Review", "Notes")
        tags = self._tags(first(row, "Tags", "Tag"))
        rewatch = self._truthy(first(row, "Rewatch", "Rewatched"))
        now = datetime.now(timezone.utc)

        metadata = clean_metadata(
            {
                "title": title,
                "year": year,
                "letterboxd_uri": uri,
                "watched_date": watched_text,
                "watched_at": watched_at.isoformat() if watched_at else None,
                "rating": rating,
                "rewatch": rewatch,
                "review": review,
                "tags": tags,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="letterboxd_watched_csv",
            source_id=digest_source_id("letterboxd_watched_csv", uri or title, year, watched_text, index if not uri else ""),
            source_entity_type="film",
            title=f"{title} ({year})" if title and year else title or uri or "Untitled Letterboxd film",
            content=self._content(title, year, uri, watched_text, rating, rewatch, review, tags),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=watched_at or now,
            updated_at=watched_at or now,
        )

    def _content(self, title: str, year: str, uri: str, watched: str, rating: float | None, rewatch: bool, review: str, tags: list[str]) -> str:
        parts = [part for part in (title, f"Year: {year}" if year else "", f"Letterboxd URI: {uri}" if uri else "") if part]
        if watched:
            parts.append(f"Watched: {watched}")
        if rating is not None:
            parts.append(f"Rating: {rating:g}")
        if rewatch:
            parts.append("Rewatch: true")
        if review:
            parts.append(f"Review: {review}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

    def _tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for tag in split_values(value):
            normalized = " ".join(tag.removeprefix("#").casefold().split())
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _truthy(self, value: str) -> bool:
        return value.strip().casefold() in {"1", "true", "yes", "y", "rewatch", "rewatched"}
