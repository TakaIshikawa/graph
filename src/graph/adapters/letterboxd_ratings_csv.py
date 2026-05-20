"""Adapter for Letterboxd ratings CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class LetterboxdRatingsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "letterboxd_ratings_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["film_rating"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "film_rating" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Name", "Film", "Title", "name", "film", "title")
        year = self._normalize_year(first(row, "Year", "Release Year", "year", "release_year"))
        rating = first(row, "Rating", "rating")
        watched_text = first(row, "Watched Date", "Date", "Rated Date", "watched_date", "date", "rated_date")
        watched_at = parse_datetime(watched_text)
        rewatch = self._parse_bool(first(row, "Rewatch", "rewatch"))
        review_url = first(row, "Review URL", "Review Url", "Review Link", "review_url", "review")
        uri = first(row, "Letterboxd URI", "Letterboxd URL", "URI", "URL", "letterboxd_uri", "letterboxd_url", "uri", "url")
        tags = split_values(first(row, "Tags", "tags"))
        if not any([title, year, rating, watched_text, rewatch is not None, review_url, uri, tags]):
            return None

        now = datetime.now(timezone.utc)
        timestamp = watched_at or now
        metadata = clean_metadata(
            {
                "title": title,
                "year": year,
                "rating": rating,
                "watched_date": watched_at.date().isoformat() if watched_at else watched_text,
                "rewatch": rewatch,
                "review_url": review_url,
                "letterboxd_uri": uri,
                "source_url": uri or review_url,
                "external_url": uri or review_url,
                "tags": tags,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.LETTERBOXD,
            source_id=self._source_id(uri, title, year, rating, source_file, index),
            source_entity_type="film_rating",
            title=self._title(title, year),
            content=self._content(title, year, rating, metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["letterboxd", "film_rating", *tags] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _source_id(self, uri: str, title: str, year: str, rating: str, source_file: str, index: int) -> str:
        if uri:
            return digest_source_id("letterboxd_ratings_csv", uri)
        return digest_source_id("letterboxd_ratings_csv", title, year, rating, source_file, index if not any([title, year, rating]) else "")

    def _title(self, title: str, year: str) -> str:
        if title and year:
            return f"{title} ({year})"
        return title or "Untitled Letterboxd rating"

    def _content(self, title: str, year: str, rating: str, metadata: dict[str, Any]) -> str:
        parts = []
        if title or year:
            parts.append(f"Film: {self._title(title, year)}")
        if rating:
            parts.append(f"Rating: {rating}")
        if metadata.get("watched_date"):
            parts.append(f"Watched date: {metadata['watched_date']}")
        if metadata.get("rewatch") is not None:
            parts.append(f"Rewatch: {metadata['rewatch']}")
        if metadata.get("tags"):
            parts.append(f"Tags: {', '.join(metadata['tags'])}")
        if metadata.get("letterboxd_uri"):
            parts.append(f"Letterboxd URI: {metadata['letterboxd_uri']}")
        if metadata.get("review_url"):
            parts.append(f"Review URL: {metadata['review_url']}")
        return "\n".join(parts)

    def _normalize_year(self, value: str) -> str:
        match = re.search(r"\d{4}", value or "")
        return match.group(0) if match else ""

    def _parse_bool(self, value: str) -> bool | None:
        text = value.casefold().strip()
        if not text:
            return None
        if text in {"1", "true", "yes", "y", "rewatch"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
        return None
