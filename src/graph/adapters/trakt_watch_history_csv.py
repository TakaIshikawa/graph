"""Adapter for Trakt watch history CSV exports."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class TraktWatchHistoryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "trakt_watch_history_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["watch"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "watch" not in entity_types:
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.created_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.created_at, unit.source_id)))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.csv") if child.is_file())

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in csv.DictReader(handle)]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        watched_text = self._first(row, "watched_at", "Watched At", "watched date", "watched")
        watched_at = self._parse_datetime(watched_text)
        title = self._first(row, "title", "Title", "movie_title", "show_title", "episode_title", "name")
        media_type = self._first(row, "type", "Type", "media_type") or "watch"
        if watched_at is None or not title:
            return None

        year = self._first(row, "year", "Year")
        season = self._parse_int(self._first(row, "season", "Season"))
        episode = self._parse_int(self._first(row, "episode", "Episode"))
        imdb_id = self._first(row, "imdb_id", "imdb", "IMDB ID")
        tmdb_id = self._first(row, "tmdb_id", "tmdb", "TMDB ID")
        trakt_id = self._first(row, "trakt_id", "trakt", "Trakt ID")
        urls = self._url_fields(row)

        metadata = {
            "watched_at": watched_at.isoformat(),
            "title": title,
            "year": year,
            "type": media_type,
            "season": season,
            "episode": episode,
            "imdb_id": imdb_id,
            "tmdb_id": tmdb_id,
            "trakt_id": trakt_id,
            "urls": urls,
            "source_file": source_file,
            "row": dict(row),
        }
        tags = ["trakt", media_type.lower()]
        return KnowledgeUnit(
            source_project=SourceProject.TRAKT_WATCH_HISTORY_CSV,
            source_id=self._source_id(row, watched_at, title),
            source_entity_type="watch",
            title=self._title(title, year, media_type, season, episode),
            content=self._content(title, year, media_type, season, episode, watched_at, imdb_id, tmdb_id, trakt_id, urls),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=tags,
            created_at=watched_at,
            updated_at=watched_at,
        )

    def _source_id(self, row: dict[str, Any], watched_at: datetime, title: str) -> str:
        explicit = self._first(row, "history_id", "id", "ID")
        raw = explicit or "|".join(
            [
                watched_at.isoformat(),
                title,
                self._first(row, "type", "Type"),
                self._first(row, "season", "Season"),
                self._first(row, "episode", "Episode"),
                self._first(row, "trakt_id", "trakt", "Trakt ID"),
            ]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"trakt_watch_history_csv:{digest}"

    def _title(self, title: str, year: str, media_type: str, season: int | None, episode: int | None) -> str:
        formatted = f"{title} ({year})" if year else title
        if media_type.lower() in {"episode", "show"} and season is not None and episode is not None:
            return f"{formatted} S{season:02d}E{episode:02d}"
        return formatted

    def _content(
        self,
        title: str,
        year: str,
        media_type: str,
        season: int | None,
        episode: int | None,
        watched_at: datetime,
        imdb_id: str,
        tmdb_id: str,
        trakt_id: str,
        urls: dict[str, str],
    ) -> str:
        parts = [f"Title: {title}", f"Watched at: {watched_at.isoformat()}", f"Type: {media_type}"]
        if year:
            parts.append(f"Year: {year}")
        if season is not None:
            parts.append(f"Season: {season}")
        if episode is not None:
            parts.append(f"Episode: {episode}")
        if imdb_id:
            parts.append(f"IMDb ID: {imdb_id}")
        if tmdb_id:
            parts.append(f"TMDb ID: {tmdb_id}")
        if trakt_id:
            parts.append(f"Trakt ID: {trakt_id}")
        for key, value in urls.items():
            parts.append(f"{key}: {value}")
        return "\n".join(parts)

    def _url_fields(self, row: dict[str, Any]) -> dict[str, str]:
        urls: dict[str, str] = {}
        for key, value in row.items():
            text = "" if value is None else str(value).strip()
            if text and any(token in str(key).lower() for token in ("url", "uri", "link")):
                urls[str(key).strip()] = text
        return urls

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).lower(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.lower())
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(float(str(value).strip()))
        except ValueError:
            return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        text = str(value).strip()
        for candidate in (text, f"{text}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y", "%m/%d/%Y %H:%M"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
