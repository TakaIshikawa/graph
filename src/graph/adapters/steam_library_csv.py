"""Adapter for Steam library CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class SteamLibraryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "steam_library_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["game", "genre", "developer"]

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

        game_units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                game_units.append(unit)

        genre_units = self._genre_units(game_units)
        developer_units = self._developer_units(game_units)
        if "game" in allowed_types:
            result.units.extend(game_units)
        if "genre" in allowed_types:
            result.units.extend(genre_units)
        if "developer" in allowed_types:
            result.units.extend(developer_units)
        if {"game", "genre"}.issubset(allowed_types):
            result.edges.extend(self._genre_edges(genre_units, game_units))
        if {"game", "developer"}.issubset(allowed_types):
            result.edges.extend(self._developer_edges(developer_units, game_units))
        result.units.sort(key=lambda unit: unit.source_id)
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
        app_id = self._first(row, "app_id", "appid", "app id", "AppID", "Application ID", "steam_appid", "Game ID")
        title = self._first(row, "title", "name", "game", "Game", "Title", "Name")
        if not app_id and not title:
            return None

        playtime = self._playtime_minutes(row)
        last_played_text = self._first(row, "last_played", "last played", "Last Played", "last_played_at", "Last Played At")
        last_played = self._parse_datetime(last_played_text)
        store_url = self._first(row, "store_url", "store url", "url", "URL", "link", "Store URL")
        if not store_url and app_id:
            store_url = f"https://store.steampowered.com/app/{app_id}/"
        platform = self._first(row, "platform", "Platform") or "steam"
        tags = self._tags(row)
        creators = self._creator_values(row)
        now = datetime.now(timezone.utc)

        metadata = {
            "app_id": app_id,
            "playtime_minutes": playtime,
            "playtime_bucket": self._playtime_bucket(playtime),
            "last_played": last_played.isoformat() if last_played else last_played_text,
            "store_url": store_url,
            "platform": platform,
            "genres": self._genre_values(row),
            "creators": creators,
            "source_file": source_file,
            "row": dict(row),
        }
        return KnowledgeUnit(
            source_project=SourceProject.STEAM_LIBRARY_CSV,
            source_id=self._source_id(app_id, title),
            source_entity_type="game",
            title=title or f"Steam app {app_id}",
            content=self._content(title or f"Steam app {app_id}", playtime, last_played, store_url, tags),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None)},
            tags=tags,
            created_at=last_played or now,
            updated_at=last_played or now,
        )

    def _source_id(self, app_id: str, title: str) -> str:
        if app_id:
            return f"steam_library_csv:{app_id}"
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:24]
        return f"steam_library_csv:{digest}"

    def _content(self, title: str, playtime: int | None, last_played: datetime | None, store_url: str, tags: list[str]) -> str:
        parts = [title]
        if playtime is not None:
            parts.append(f"Playtime: {playtime} minutes")
        if last_played is not None:
            parts.append(f"Last played: {last_played.isoformat()}")
        if store_url:
            parts.append(f"Store URL: {store_url}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

    def _tags(self, row: dict[str, Any]) -> list[str]:
        tags = ["steam", "game"]
        for tag in self._genre_values(row):
            if tag not in tags:
                tags.append(tag)
        return tags

    def _genre_values(self, row: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("Genres", "genres", "Categories", "categories", "Tags", "tags"):
            for value in re.split(r"[,;|]", self._first(row, key)):
                tag = " ".join(value.strip().casefold().split())
                if tag and tag not in values:
                    values.append(tag)
        return values

    def _creator_values(self, row: dict[str, Any]) -> list[dict[str, str]]:
        creators: list[dict[str, str]] = []
        seen: set[str] = set()
        for key in ("Developer", "Developers", "Publisher", "Publishers"):
            for value in re.split(r"[,;|]", self._first(row, key)):
                name = " ".join(value.strip().split())
                normalized = " ".join(name.casefold().split())
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                creators.append({"name": name, "normalized_name": normalized})
        return creators

    def _genre_units(self, games: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for game in games:
            for genre in game.metadata.get("genres") or []:
                grouped.setdefault(str(genre), []).append(game)

        units: list[KnowledgeUnit] = []
        for genre, genre_games in sorted(grouped.items()):
            playtimes = [value for game in genre_games if (value := game.metadata.get("playtime_minutes")) is not None]
            last_played = [
                parsed
                for game in genre_games
                if (parsed := self._parse_datetime(game.metadata.get("last_played"))) is not None
            ]
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.STEAM_LIBRARY_CSV,
                    source_id=self._genre_source_id(genre),
                    source_entity_type="genre",
                    title=genre,
                    content=f"Steam genre: {genre}\nGames: {len(genre_games)}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "genre": genre,
                        "game_count": len(genre_games),
                        "total_playtime_minutes": sum(playtimes),
                        "game_source_ids": sorted(game.source_id for game in genre_games),
                        "app_ids": sorted(str(game.metadata.get("app_id")) for game in genre_games if game.metadata.get("app_id")),
                        "last_played_at": max(last_played).isoformat() if last_played else None,
                        "source_files": sorted({str(game.metadata.get("source_file")) for game in genre_games if game.metadata.get("source_file")}),
                    },
                    tags=["steam", "genre", genre],
                    created_at=min(game.created_at for game in genre_games),
                    updated_at=max(game.updated_at for game in genre_games),
                )
            )
        return units

    def _genre_edges(self, genres: list[KnowledgeUnit], games: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        genre_ids = {str(genre.metadata.get("genre")): genre.source_id for genre in genres}
        edges: list[KnowledgeEdge] = []
        for game in games:
            for genre in game.metadata.get("genres") or []:
                genre_id = genre_ids.get(str(genre))
                if not genre_id:
                    continue
                digest = hashlib.sha256("|".join((genre_id, game.source_id, "genre_contains_game")).encode("utf-8")).hexdigest()[:24]
                edges.append(
                    KnowledgeEdge(
                        id=f"steam-library-csv-genre-contains-{digest}",
                        from_unit_id=genre_id,
                        to_unit_id=game.source_id,
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={"source_project": SourceProject.STEAM_LIBRARY_CSV.value, "relation_type": "genre_contains_game"},
                    )
                )
        return edges

    def _genre_source_id(self, genre: str) -> str:
        digest = hashlib.sha256(genre.encode("utf-8")).hexdigest()[:24]
        return f"steam_library_csv:genre:{digest}"

    def _developer_units(self, games: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for game in games:
            for creator in game.metadata.get("creators") or []:
                normalized = str(creator.get("normalized_name") or "")
                if not normalized:
                    continue
                grouped.setdefault(normalized, []).append(game)
                names.setdefault(normalized, str(creator.get("name") or normalized))

        units: list[KnowledgeUnit] = []
        for normalized, creator_games in sorted(grouped.items()):
            unique_games = sorted({game.source_id: game for game in creator_games}.values(), key=lambda game: game.source_id)
            playtimes = [value for game in unique_games if (value := game.metadata.get("playtime_minutes")) is not None]
            app_ids = sorted(str(game.metadata.get("app_id")) for game in unique_games if game.metadata.get("app_id"))
            name = names[normalized]
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.STEAM_LIBRARY_CSV,
                    source_id=self._developer_source_id(normalized),
                    source_entity_type="developer",
                    title=name,
                    content=f"Steam creator: {name}\nGames: {len(unique_games)}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "developer": name,
                        "normalized_name": normalized,
                        "game_count": len(unique_games),
                        "total_playtime_minutes": sum(playtimes),
                        "game_source_ids": [game.source_id for game in unique_games],
                        "app_ids": app_ids,
                        "source_files": sorted({str(game.metadata.get("source_file")) for game in unique_games if game.metadata.get("source_file")}),
                    },
                    tags=["steam", "developer", name],
                    created_at=min(game.created_at for game in unique_games),
                    updated_at=max(game.updated_at for game in unique_games),
                )
            )
        return units

    def _developer_edges(self, developers: list[KnowledgeUnit], games: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        developer_ids = {str(developer.metadata.get("normalized_name")): developer.source_id for developer in developers}
        edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str]] = set()
        for game in games:
            for creator in game.metadata.get("creators") or []:
                developer_id = developer_ids.get(str(creator.get("normalized_name") or ""))
                if not developer_id or (developer_id, game.source_id) in seen:
                    continue
                seen.add((developer_id, game.source_id))
                digest = hashlib.sha256("|".join((developer_id, game.source_id, "developer_contains_game")).encode("utf-8")).hexdigest()[:24]
                edges.append(
                    KnowledgeEdge(
                        id=f"steam-library-csv-developer-contains-{digest}",
                        from_unit_id=developer_id,
                        to_unit_id=game.source_id,
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.STEAM_LIBRARY_CSV.value,
                            "relation_type": "developer_contains_game",
                            "developer": creator.get("name"),
                        },
                    )
                )
        return edges

    def _developer_source_id(self, normalized_name: str) -> str:
        digest = hashlib.sha256(normalized_name.encode("utf-8")).hexdigest()[:24]
        return f"steam_library_csv:developer:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).casefold(): value for key, value in row.items()}
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.casefold())
            if value is None:
                value = compact.get(self._normalize_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _parse_playtime_minutes(self, value: str) -> int | None:
        if not value:
            return None
        text = value.strip().lower().replace(",", "")
        try:
            number = float(re.search(r"-?\d+(?:\.\d+)?", text).group(0))  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            return None
        if number < 0:
            return None
        if "hour" in text or re.search(r"\bhrs?\b", text):
            return int(round(number * 60))
        return int(round(number))

    def _playtime_minutes(self, row: dict[str, Any]) -> int | None:
        minute_value = self._first(row, "playtime_minutes", "playtime_forever", "minutes", "Minutes Played")
        if minute_value:
            return self._parse_playtime_minutes(minute_value)
        hour_value = self._first(row, "hours", "hours_played", "Hours Played", "Playtime Hours")
        if hour_value:
            text = hour_value.strip().lower()
            if re.search(r"hour|\bhrs?\b", text):
                return self._parse_playtime_minutes(hour_value)
            try:
                match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
                if not match:
                    return None
                number = float(match.group(0))
                return int(round(number * 60)) if number >= 0 else None
            except ValueError:
                return None
        return self._parse_playtime_minutes(self._first(row, "playtime", "Playtime"))

    def _playtime_bucket(self, minutes: int | None) -> str | None:
        if minutes is None:
            return None
        if minutes <= 0:
            return "unplayed"
        if minutes < 60:
            return "sampled"
        if minutes < 600:
            return "played"
        return "deep_play"

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        text = value.strip()
        for candidate in (text, f"{text}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y", "%m/%d/%Y %H:%M", "%b %d, %Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
