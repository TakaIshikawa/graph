"""Adapter for Letterboxd diary.csv exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
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

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or self.entity_types)
        if not allowed.intersection(self.entity_types):
            return result
        sync_at = self._sync_datetime(since) if since else None

        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _iter_paths(self) -> list[Path]:
        path = Path(self.path).expanduser() if self.path else None
        if path is None:
            return []
        if path.is_file() and path.suffix.lower() == ".csv":
            return [path]
        if path.is_dir():
            return sorted(child for child in path.rglob("*.csv") if child.is_file())
        return []

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in csv.DictReader(handle)]

    def _unit_from_row(self, row: dict[str, Any], path: Path, row_index: int) -> KnowledgeUnit | None:
        title = self._first(row, "Name", "Film", "Title", "name", "film", "title")
        if not title:
            return None

        year = self._first(row, "Year", "Release Year", "year", "release_year")
        watched_text = self._first(row, "Watched Date", "Date", "watched_date", "date")
        watched_at = self._parse_datetime(watched_text)
        rating = self._first(row, "Rating", "rating")
        rewatch = self._parse_bool(self._first(row, "Rewatch", "rewatch"))
        tags = self._tags(self._first(row, "Tags", "tags"))
        review = self._first(row, "Review", "review")
        uri = self._first(row, "Letterboxd URI", "Letterboxd URL", "URI", "URL", "letterboxd_uri", "url")
        now = datetime.now(timezone.utc)

        metadata = {
            "title": title,
            "year": self._normalize_year(year),
            "watched_date": watched_at.date().isoformat() if watched_at else watched_text,
            "rating": rating,
            "rewatch": rewatch,
            "tags": tags,
            "review": review,
            "letterboxd_uri": uri,
            "source_file": str(path),
            "row_index": row_index,
            "row": dict(row),
        }
        return KnowledgeUnit(
            source_project=SourceProject.LETTERBOXD,
            source_id=self._source_id(uri, title, metadata["year"], metadata["watched_date"], row_index),
            source_entity_type="diary_entry",
            title=f"{title} ({metadata['year']})" if metadata["year"] else title,
            content=self._content(title, metadata["year"], metadata["watched_date"], rating, review, tags),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=watched_at or now,
            updated_at=watched_at or now,
        )

    def _content(self, title: str, year: str, watched_date: str, rating: str, review: str, tags: list[str]) -> str:
        parts = [f"Film: {title} ({year})" if year else f"Film: {title}"]
        if watched_date:
            parts.append(f"Watched date: {watched_date}")
        if rating:
            parts.append(f"Rating: {rating}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if review:
            parts.append(f"\nReview:\n{review}")
        return "\n".join(parts)

    def _source_id(self, uri: str, title: str, year: str, watched_date: str, row_index: int) -> str:
        raw = "|".join((uri, title, year, watched_date, str(row_index)))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"letterboxd_diary_csv:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _normalize_year(self, value: str) -> str:
        match = re.search(r"\d{4}", value or "")
        return match.group(0) if match else ""

    def _tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for item in re.split(r"[,;|]", value or ""):
            tag = item.strip()
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _parse_bool(self, value: str) -> bool | None:
        text = value.casefold().strip()
        if not text:
            return None
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
        return None

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"):
                try:
                    parsed = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
