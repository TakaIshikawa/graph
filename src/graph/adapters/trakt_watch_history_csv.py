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
        return ["media", "watch", "rating", "show", "season", "watched_day"]

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
        watches = [unit for unit in units if unit.source_entity_type == "watch"]
        ratings = [unit for unit in units if unit.source_entity_type == "rating"]
        self._add_rewatch_metadata(watches)
        media = self._media_units(units) if "media" in allowed_types else []
        shows = self._show_units(watches) if "show" in allowed_types else []
        seasons = self._season_units(watches) if "season" in allowed_types else []
        watched_days = self._watched_day_units(watches) if "watched_day" in allowed_types else []
        if "media" in allowed_types:
            result.units.extend(media)
        if "show" in allowed_types:
            result.units.extend(shows)
        if "season" in allowed_types:
            result.units.extend(seasons)
        if "watch" in allowed_types:
            result.units.extend(watches)
        if "rating" in allowed_types:
            result.units.extend(ratings)
        if "watched_day" in allowed_types:
            result.units.extend(watched_days)
        if "media" in allowed_types and "watch" in allowed_types:
            result.edges.extend(self._media_item_edges(media, watches, "media_contains_watch"))
        if "media" in allowed_types and "rating" in allowed_types:
            result.edges.extend(self._media_item_edges(media, ratings, "media_contains_rating"))
        if "show" in allowed_types and "season" in allowed_types:
            result.edges.extend(self._show_season_edges(shows, seasons))
        if "season" in allowed_types and "watch" in allowed_types:
            result.edges.extend(self._season_watch_edges(seasons, watches))
        if "watched_day" in allowed_types and "watch" in allowed_types:
            result.edges.extend(self._watched_day_watch_edges(watched_days))
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
        rated_text = self._first(row, "rated_at", "Rated At", "rated date", "rated")
        rated_at = self._parse_datetime(rated_text)
        title = self._first(row, "title", "Title", "movie_title", "show_title", "episode_title", "name")
        media_type = self._first(row, "type", "Type", "media_type") or "watch"
        is_rating = self._is_rating_row(row, rated_at)
        event_at = rated_at if is_rating else watched_at
        if event_at is None or not title:
            return None

        year = self._first(row, "year", "Year")
        season = self._parse_int(self._first(row, "season", "Season"))
        episode = self._parse_int(self._first(row, "episode", "Episode"))
        imdb_id = self._first(row, "imdb_id", "imdb", "IMDB ID")
        tmdb_id = self._first(row, "tmdb_id", "tmdb", "TMDB ID")
        trakt_id = self._first(row, "trakt_id", "trakt", "Trakt ID")
        urls = self._url_fields(row)
        show_title = self._first(row, "show_title", "Show Title", "show", "Show") or title
        episode_title = self._first(row, "episode_title", "Episode Title")
        show_year = self._first(row, "show_year", "Show Year") or year
        show_url = self._show_url(urls)
        rating = self._parse_int(self._first(row, "rating", "Rating", "rated", "Rated"))

        metadata = {
            "title": title,
            "year": year,
            "type": media_type,
            "show_title": show_title,
            "episode_title": episode_title,
            "show_year": show_year,
            "show_url": show_url,
            "season": season,
            "episode": episode,
            "imdb_id": imdb_id,
            "tmdb_id": tmdb_id,
            "trakt_id": trakt_id,
            "urls": urls,
            "source_file": source_file,
            "row": dict(row),
        }
        if is_rating:
            metadata["rated_at"] = event_at.isoformat()
            metadata["rating"] = rating
            metadata["watched_at"] = watched_at.isoformat() if watched_at else ""
        else:
            metadata["watched_at"] = event_at.isoformat()
        tags = ["trakt", media_type.lower()]
        entity_type = "rating" if is_rating else "watch"
        return KnowledgeUnit(
            source_project=SourceProject.TRAKT_WATCH_HISTORY_CSV,
            source_id=self._source_id(row, event_at, title, entity_type),
            source_entity_type=entity_type,
            title=self._title(title, year, media_type, season, episode),
            content=self._content(title, year, media_type, season, episode, event_at, imdb_id, tmdb_id, trakt_id, urls, rating if is_rating else None),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=tags,
            created_at=event_at,
            updated_at=event_at,
        )

    def _is_rating_row(self, row: dict[str, Any], rated_at: datetime | None) -> bool:
        if rated_at is not None:
            return True
        return bool(self._first(row, "rating", "Rating")) and not self._first(row, "watched_at", "Watched At", "watched date", "watched")

    def _source_id(self, row: dict[str, Any], event_at: datetime, title: str, entity_type: str) -> str:
        explicit = self._first(row, "history_id", "rating_id", "id", "ID")
        raw = explicit or "|".join(
            [
                entity_type,
                event_at.isoformat(),
                title,
                self._first(row, "type", "Type"),
                self._first(row, "season", "Season"),
                self._first(row, "episode", "Episode"),
                self._first(row, "trakt_id", "trakt", "Trakt ID"),
                self._first(row, "rating", "Rating", "rated"),
            ]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"trakt_watch_history_csv:{entity_type}:{digest}"

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

    def _media_units(self, items: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[tuple[Any, ...], list[KnowledgeUnit]] = {}
        for item in items:
            grouped.setdefault(self._watch_identity(item.metadata), []).append(item)

        units: list[KnowledgeUnit] = []
        for identity, media_items in grouped.items():
            first = media_items[0]
            identifiers = {
                key: first.metadata.get(key)
                for key in ("trakt_id", "imdb_id", "tmdb_id")
                if first.metadata.get(key)
            }
            urls: dict[str, str] = {}
            for item in media_items:
                urls.update(item.metadata.get("urls") or {})
            watches = [item for item in media_items if item.source_entity_type == "watch"]
            ratings = [item for item in media_items if item.source_entity_type == "rating"]
            first_seen = min(item.created_at for item in media_items)
            last_seen = max(item.created_at for item in media_items)
            first_watched = min((watch.created_at for watch in watches), default=None)
            last_watched = max((watch.created_at for watch in watches), default=None)
            first_rated = min((rating.created_at for rating in ratings), default=None)
            last_rated = max((rating.created_at for rating in ratings), default=None)
            rewatch_count = sum(1 for watch in watches if watch.metadata.get("is_rewatch"))
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
                        "watch_count": len(watches),
                        "rating_count": len(ratings),
                        "rewatch_count": rewatch_count,
                        "first_watched_at": first_watched.isoformat() if first_watched else None,
                        "last_watched_at": last_watched.isoformat() if last_watched else None,
                        "first_rated_at": first_rated.isoformat() if first_rated else None,
                        "last_rated_at": last_rated.isoformat() if last_rated else None,
                        "ratings": [rating.metadata.get("rating") for rating in ratings if rating.metadata.get("rating") is not None],
                        "identifiers": identifiers,
                        "urls": urls,
                        "source_files": sorted({str(item.metadata.get("source_file")) for item in media_items if item.metadata.get("source_file")}),
                        "watch_source_ids": [watch.source_id for watch in watches],
                        "rating_source_ids": [rating.source_id for rating in ratings],
                    },
                    tags=["trakt", str(first.metadata.get("type") or "media").lower(), "media"],
                    created_at=first_seen,
                    updated_at=last_seen,
                )
            )
        return units

    def _show_units(self, watches: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[tuple[Any, ...], list[KnowledgeUnit]] = {}
        for watch in watches:
            key = self._show_identity(watch.metadata)
            if key:
                grouped.setdefault(key, []).append(watch)

        units: list[KnowledgeUnit] = []
        for identity, show_watches in grouped.items():
            first = show_watches[0]
            seasons = sorted({watch.metadata.get("season") for watch in show_watches if watch.metadata.get("season") is not None})
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.TRAKT_WATCH_HISTORY_CSV,
                    source_id=self._show_source_id(identity),
                    source_entity_type="show",
                    title=self._show_title(first.metadata),
                    content=f"Trakt show: {self._show_title(first.metadata)}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "type": "episode",
                        "show_title": first.metadata.get("show_title"),
                        "show_year": first.metadata.get("show_year"),
                        "show_url": first.metadata.get("show_url"),
                        "season_numbers": seasons,
                        "season_count": len(seasons),
                        "watch_count": len(show_watches),
                        "first_watched_at": min(watch.created_at for watch in show_watches).isoformat(),
                        "last_watched_at": max(watch.created_at for watch in show_watches).isoformat(),
                        "watch_source_ids": [watch.source_id for watch in show_watches],
                        "source_files": sorted({str(watch.metadata.get("source_file")) for watch in show_watches if watch.metadata.get("source_file")}),
                    },
                    tags=["trakt", "show"],
                    created_at=min(watch.created_at for watch in show_watches),
                    updated_at=max(watch.created_at for watch in show_watches),
                )
            )
        return units

    def _season_units(self, watches: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[tuple[Any, ...], list[KnowledgeUnit]] = {}
        for watch in watches:
            key = self._season_identity(watch.metadata)
            if key:
                grouped.setdefault(key, []).append(watch)

        units: list[KnowledgeUnit] = []
        for identity, season_watches in grouped.items():
            first = season_watches[0]
            season = first.metadata.get("season")
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.TRAKT_WATCH_HISTORY_CSV,
                    source_id=self._season_source_id(identity),
                    source_entity_type="season",
                    title=f"{self._show_title(first.metadata)} Season {season}",
                    content=f"Trakt season: {self._show_title(first.metadata)} Season {season}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "type": "episode",
                        "show_title": first.metadata.get("show_title"),
                        "show_year": first.metadata.get("show_year"),
                        "show_url": first.metadata.get("show_url"),
                        "season": season,
                        "episode_numbers": sorted({watch.metadata.get("episode") for watch in season_watches if watch.metadata.get("episode") is not None}),
                        "watch_count": len(season_watches),
                        "first_watched_at": min(watch.created_at for watch in season_watches).isoformat(),
                        "last_watched_at": max(watch.created_at for watch in season_watches).isoformat(),
                        "watch_source_ids": [watch.source_id for watch in season_watches],
                        "source_files": sorted({str(watch.metadata.get("source_file")) for watch in season_watches if watch.metadata.get("source_file")}),
                    },
                    tags=["trakt", "season"],
                    created_at=min(watch.created_at for watch in season_watches),
                    updated_at=max(watch.created_at for watch in season_watches),
                )
            )
        return units

    def _watched_day_units(self, watches: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for watch in watches:
            grouped.setdefault(watch.created_at.date().isoformat(), []).append(watch)

        units: list[KnowledgeUnit] = []
        for date, day_watches in sorted(grouped.items()):
            first_seen = min(watch.created_at for watch in day_watches)
            last_seen = max(watch.created_at for watch in day_watches)
            media_identities = {self._watch_identity(watch.metadata) for watch in day_watches}
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.TRAKT_WATCH_HISTORY_CSV,
                    source_id=self._watched_day_source_id(date),
                    source_entity_type="watched_day",
                    title=f"Trakt watches on {date}",
                    content=f"{len(day_watches)} Trakt watches on {date}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "date": date,
                        "watch_count": len(day_watches),
                        "movie_count": sum(1 for watch in day_watches if str(watch.metadata.get("type") or "").casefold() == "movie"),
                        "episode_count": sum(1 for watch in day_watches if str(watch.metadata.get("type") or "").casefold() == "episode"),
                        "rewatch_count": sum(1 for watch in day_watches if watch.metadata.get("is_rewatch")),
                        "unique_media_count": len(media_identities),
                        "source_files": sorted({str(watch.metadata.get("source_file")) for watch in day_watches if watch.metadata.get("source_file")}),
                        "watch_source_ids": [watch.source_id for watch in day_watches],
                    },
                    tags=["trakt", "watched-day"],
                    created_at=first_seen,
                    updated_at=last_seen,
                )
            )
        return units

    def _media_item_edges(self, media: list[KnowledgeUnit], items: list[KnowledgeUnit], relation_type: str) -> list[KnowledgeEdge]:
        media_ids = {}
        for unit in media:
            identity = self._media_identity_from_metadata(unit.metadata)
            media_ids[identity] = unit.source_id
        edges: list[KnowledgeEdge] = []
        for item in items:
            media_id = media_ids.get(self._watch_identity(item.metadata))
            if not media_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=self._edge_id(media_id, item.source_id, relation_type),
                    from_unit_id=media_id,
                    to_unit_id=item.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.TRAKT_WATCH_HISTORY_CSV.value,
                        "relation_type": relation_type,
                    },
                )
            )
        return list({edge.id: edge for edge in edges}.values())

    def _show_season_edges(self, shows: list[KnowledgeUnit], seasons: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        show_ids = {self._show_identity(unit.metadata): unit.source_id for unit in shows}
        edges: list[KnowledgeEdge] = []
        for season in seasons:
            show_id = show_ids.get(self._show_identity(season.metadata))
            if show_id:
                edges.append(self._edge(show_id, season.source_id, "show_contains_season"))
        return list({edge.id: edge for edge in edges}.values())

    def _season_watch_edges(self, seasons: list[KnowledgeUnit], watches: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        season_ids = {self._season_identity(unit.metadata): unit.source_id for unit in seasons}
        edges: list[KnowledgeEdge] = []
        for watch in watches:
            season_id = season_ids.get(self._season_identity(watch.metadata))
            if season_id:
                edges.append(self._edge(season_id, watch.source_id, "season_contains_watch"))
        return list({edge.id: edge for edge in edges}.values())

    def _watched_day_watch_edges(self, watched_days: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for day in watched_days:
            for watch_source_id in day.metadata.get("watch_source_ids") or []:
                edges.append(self._edge(day.source_id, str(watch_source_id), "watched_day_contains_watch"))
        return list({edge.id: edge for edge in edges}.values())

    def _edge(self, from_id: str, to_id: str, relation_type: str) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=self._edge_id(from_id, to_id, relation_type),
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.TRAKT_WATCH_HISTORY_CSV.value,
                "relation_type": relation_type,
            },
        )

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

    def _show_identity(self, metadata: dict[str, Any]) -> tuple[Any, ...] | None:
        if str(metadata.get("type") or "").casefold() not in {"episode", "show"}:
            return None
        show_url = str(metadata.get("show_url") or "").strip()
        if show_url:
            return ("url", show_url.casefold())
        show_title = self._normalize_identity_text(metadata.get("show_title") or metadata.get("title"))
        if not show_title:
            return None
        return ("fallback", show_title, self._normalize_identity_text(metadata.get("show_year") or metadata.get("year")))

    def _season_identity(self, metadata: dict[str, Any]) -> tuple[Any, ...] | None:
        show_identity = self._show_identity(metadata)
        season = metadata.get("season")
        if show_identity is None or season is None:
            return None
        return (*show_identity, "season", season)

    def _show_source_id(self, identity: tuple[Any, ...]) -> str:
        digest = hashlib.sha256("|".join(str(part) for part in identity).encode("utf-8")).hexdigest()[:24]
        return f"trakt_watch_history_csv:show:{digest}"

    def _season_source_id(self, identity: tuple[Any, ...]) -> str:
        digest = hashlib.sha256("|".join(str(part) for part in identity).encode("utf-8")).hexdigest()[:24]
        return f"trakt_watch_history_csv:season:{digest}"

    def _watched_day_source_id(self, date: str) -> str:
        digest = hashlib.sha256(date.encode("utf-8")).hexdigest()[:24]
        return f"trakt_watch_history_csv:watched_day:{digest}"

    def _show_title(self, metadata: dict[str, Any]) -> str:
        title = str(metadata.get("show_title") or metadata.get("title") or "")
        year = str(metadata.get("show_year") or metadata.get("year") or "")
        return f"{title} ({year})" if year else title

    def _edge_id(self, media_source_id: str, watch_source_id: str, relation_type: str = "media_contains_watch") -> str:
        digest = hashlib.sha256("|".join((media_source_id, watch_source_id, relation_type)).encode("utf-8")).hexdigest()[:24]
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
        event_at: datetime,
        imdb_id: str,
        tmdb_id: str,
        trakt_id: str,
        urls: dict[str, str],
        rating: int | None = None,
    ) -> str:
        label = "Rated at" if rating is not None else "Watched at"
        parts = [f"Title: {title}", f"{label}: {event_at.isoformat()}", f"Type: {media_type}"]
        if rating is not None:
            parts.append(f"Rating: {rating}")
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

    def _show_url(self, urls: dict[str, str]) -> str:
        for value in urls.values():
            if "/shows/" not in value:
                continue
            return value.split("/seasons/", 1)[0].split("/episodes/", 1)[0].rstrip("/")
        return ""

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
