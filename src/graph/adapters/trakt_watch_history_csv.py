"""Adapter for Trakt watch history CSV exports."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class TraktWatchHistoryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "trakt_watch_history_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["media", "watch"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
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

        units = sorted(units, key=lambda unit: (unit.created_at, unit.source_id))
        self._add_rewatch_metadata(units)
        media = self._media_units(units) if "media" in allowed_types else []
        if "media" in allowed_types:
            result.units.extend(media)
        if "watch" in allowed_types:
            result.units.extend(units)
        if "media" in allowed_types and "watch" in allowed_types:
            result.edges.extend(self._media_watch_edges(media, units))
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
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

    def _add_rewatch_metadata(self, units: list[KnowledgeUnit]) -> None:
        previous_by_identity: dict[tuple[Any, ...], datetime] = {}
        sequence_by_identity: dict[tuple[Any, ...], int] = {}
        for unit in units:
            identity = self._watch_identity(unit.metadata)
            sequence = sequence_by_identity.get(identity, 0) + 1
            sequence_by_identity[identity] = sequence
            unit.metadata["watch_sequence"] = sequence
            unit.metadata["is_rewatch"] = sequence > 1
            previous_watch_at = previous_by_identity.get(identity)
            if previous_watch_at is not None:
                unit.metadata["previous_watch_at"] = previous_watch_at.isoformat()
            previous_by_identity[identity] = unit.created_at

    def _watch_identity(self, metadata: dict[str, Any]) -> tuple[Any, ...]:
        for key in ("trakt_id", "imdb_id", "tmdb_id"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return (key, value.lower())
        return (
            "fallback",
            self._normalize_identity_text(metadata.get("title")),
            self._normalize_identity_text(metadata.get("year")),
            self._normalize_identity_text(metadata.get("type")),
            metadata.get("season"),
            metadata.get("episode"),
        )

    def _media_units(self, watches: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[tuple[Any, ...], list[KnowledgeUnit]] = {}
        for watch in watches:
            grouped.setdefault(self._watch_identity(watch.metadata), []).append(watch)

        units: list[KnowledgeUnit] = []
        for identity, media_watches in grouped.items():
            first = media_watches[0]
            identifiers = {
                key: first.metadata.get(key)
                for key in ("trakt_id", "imdb_id", "tmdb_id")
                if first.metadata.get(key)
            }
            urls: dict[str, str] = {}
            for watch in media_watches:
                urls.update(watch.metadata.get("urls") or {})
            first_watched = min(watch.created_at for watch in media_watches)
            last_watched = max(watch.created_at for watch in media_watches)
            rewatch_count = sum(1 for watch in media_watches if watch.metadata.get("is_rewatch"))
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.TRAKT_WATCH_HISTORY_CSV,
                    source_id=self._media_source_id(identity),
                    source_entity_type="media",
                    title=first.title,
                    content=f"Trakt media: {first.title}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "title": first.metadata.get("title"),
                        "year": first.metadata.get("year"),
                        "type": first.metadata.get("type"),
                        "season": first.metadata.get("season"),
                        "episode": first.metadata.get("episode"),
                        "watch_count": len(media_watches),
                        "rewatch_count": rewatch_count,
                        "first_watched_at": first_watched.isoformat(),
                        "last_watched_at": last_watched.isoformat(),
                        "identifiers": identifiers,
                        "urls": urls,
                        "source_files": sorted({str(watch.metadata.get("source_file")) for watch in media_watches if watch.metadata.get("source_file")}),
                        "watch_source_ids": [watch.source_id for watch in media_watches],
                    },
                    tags=["trakt", str(first.metadata.get("type") or "media").lower(), "media"],
                    created_at=first_watched,
                    updated_at=last_watched,
                )
            )
        return units

    def _media_watch_edges(self, media: list[KnowledgeUnit], watches: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        media_ids = {}
        for unit in media:
            identity = self._media_identity_from_metadata(unit.metadata)
            media_ids[identity] = unit.source_id
        edges: list[KnowledgeEdge] = []
        for watch in watches:
            media_id = media_ids.get(self._watch_identity(watch.metadata))
            if not media_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=self._edge_id(media_id, watch.source_id),
                    from_unit_id=media_id,
                    to_unit_id=watch.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.TRAKT_WATCH_HISTORY_CSV.value,
                        "relation_type": "media_contains_watch",
                    },
                )
            )
        return list({edge.id: edge for edge in edges}.values())

    def _media_identity_from_metadata(self, metadata: dict[str, Any]) -> tuple[Any, ...]:
        for key, value in (metadata.get("identifiers") or {}).items():
            if value:
                return (key, str(value).lower())
        return (
            "fallback",
            self._normalize_identity_text(metadata.get("title")),
            self._normalize_identity_text(metadata.get("year")),
            self._normalize_identity_text(metadata.get("type")),
            metadata.get("season"),
            metadata.get("episode"),
        )

    def _media_source_id(self, identity: tuple[Any, ...]) -> str:
        digest = hashlib.sha256("|".join(str(part) for part in identity).encode("utf-8")).hexdigest()[:24]
        return f"trakt_watch_history_csv:media:{digest}"

    def _edge_id(self, media_source_id: str, watch_source_id: str) -> str:
        digest = hashlib.sha256("|".join((media_source_id, watch_source_id, "contains")).encode("utf-8")).hexdigest()[:24]
        return f"trakt-watch-history-csv-contains-{digest}"

    def _normalize_identity_text(self, value: Any) -> str:
        return " ".join(str(value or "").strip().casefold().split())

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
