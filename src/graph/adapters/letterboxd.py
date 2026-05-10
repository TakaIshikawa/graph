"""Adapter for Letterboxd film diary and reviews exports."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class LetterboxdAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "letterboxd"

    @property
    def entity_types(self) -> list[str]:
        return ["film"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "film" not in entity_types:
            return result

        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists() or not path.is_file():
            return result

        try:
            items = self._read_items(path)
        except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError):
            return result

        sync_at = self._sync_datetime(since) if since else None
        for item in items:
            name = self._first(item, "Name", "name", "title", "film_name")
            if not name:
                continue

            # Use Watched Date as primary timestamp, fallback to Date
            watched_date = self._parse_datetime(
                self._first(item, "Watched Date", "watched_date", "Date", "date")
            )
            if sync_at and watched_date and watched_date <= sync_at:
                continue

            year = self._first(item, "Year", "year", "release_year")
            letterboxd_uri = self._first(item, "Letterboxd URI", "letterboxd_uri", "uri", "url")
            rating = self._first(item, "Rating", "rating")
            rewatch = self._first(item, "Rewatch", "rewatch")
            tags = self._tags(item)
            review = self._first(item, "Review", "review")

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.LETTERBOXD,
                    source_id=self._source_id(letterboxd_uri, name, year),
                    source_entity_type="film",
                    title=self._format_title(name, year),
                    content=self._content(name, year, rating, review, tags),
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "name": name,
                        "year": year,
                        "letterboxd_uri": letterboxd_uri,
                        "rating": rating,
                        "rewatch": rewatch,
                        "tags": tags,
                        "watched_date": self._first(
                            item, "Watched Date", "watched_date", "Date", "date"
                        ),
                        "review": review,
                    },
                    tags=tags,
                    created_at=watched_date or datetime.now(timezone.utc),
                    updated_at=watched_date or datetime.now(timezone.utc),
                )
            )

        return result

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                return [
                    {str(key).strip(): value for key, value in row.items() if key is not None}
                    for row in csv.DictReader(handle)
                ]

        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._json_items(parsed)

    def _json_items(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []

        for key in ("films", "diary", "reviews", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                return [item for item in nested.values() if isinstance(item, dict)]

        if any(isinstance(item, dict) for item in value.values()):
            return [item for item in value.values() if isinstance(item, dict)]
        return [value]

    def _source_id(self, letterboxd_uri: str, name: str, year: str) -> str:
        if letterboxd_uri:
            # Extract film slug from URI
            slug = letterboxd_uri.rstrip("/").split("/")[-1]
            if slug:
                return f"letterboxd:{slug}"
        identifier = f"{name}|{year}" if year else name
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
        return f"letterboxd:{digest[:24]}"

    def _format_title(self, name: str, year: str) -> str:
        if name and year:
            return f"{name} ({year})"
        return name or "Untitled Letterboxd film"

    def _content(self, name: str, year: str, rating: str, review: str, tags: list[str]) -> str:
        parts = []
        if name:
            title_part = f"{name} ({year})" if year else name
            parts.append(f"Film: {title_part}")
        if rating:
            # Letterboxd ratings are typically out of 5, often with half stars
            parts.append(f"Rating: {rating}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if review:
            parts.append(f"\nReview:\n{review}")
        return "\n".join(parts)

    def _tags(self, item: dict[str, Any]) -> list[str]:
        # Letterboxd exports tags in a "Tags" column, comma-separated
        tags_str = self._first(item, "Tags", "tags")
        if not tags_str:
            return []

        tags: list[str] = []
        for tag in re.split(r",", tags_str):
            normalized = tag.strip().lower()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        # Handle Unix timestamp
        if re.fullmatch(r"\d+(?:\.0+)?", value):
            try:
                return datetime.fromtimestamp(int(float(value)), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        # Handle ISO format and Letterboxd format (YYYY-MM-DD)
        try:
            # Try ISO format first
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            # Try common date formats
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"):
                try:
                    parsed = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
