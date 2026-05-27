"""Adapter for Letterboxd diary CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, parse_float, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class LetterboxdDiaryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "letterboxd_diary_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["diary_entry"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "diary_entry" not in entity_types:
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
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        film = first(row, "name", "film", "title")
        watched = first(row, "watched date", "date", "watched")
        year = first(row, "year", "release year")
        if not film:
            return None
        rating = parse_float(first(row, "rating"))
        rewatch = _bool(first(row, "rewatch", "re-watched"))
        tags = split_values(first(row, "tags"))
        review = first(row, "review", "review text")
        watched_at = parse_datetime(watched) or datetime.now(timezone.utc)
        title = f"{film} ({year})" if year else film
        metadata = clean_metadata({"film": film, "year": year, "watched_date": watched, "rating": rating, "rewatch": rewatch, "tags": tags, "review": review, "source_file": source_file})
        return KnowledgeUnit(source_project="letterboxd_diary_csv", source_id=digest_source_id("letterboxd_diary_csv", watched, film, year, index), source_entity_type="diary_entry", title=title, content=_content(title, watched, rating, rewatch, review, tags), content_type=ContentType.ARTIFACT, metadata=metadata, tags=tags, created_at=watched_at, updated_at=watched_at)


def _bool(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "y", "rewatch", "watched"}


def _content(title: str, watched: str, rating: float | None, rewatch: bool, review: str, tags: list[str]) -> str:
    parts = [title]
    if watched:
        parts.append(f"Watched: {watched}")
    if rating is not None:
        parts.append(f"Rating: {rating:g}")
    if rewatch:
        parts.append("Rewatch: true")
    if review:
        parts.append(review)
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    return "\n".join(parts)
