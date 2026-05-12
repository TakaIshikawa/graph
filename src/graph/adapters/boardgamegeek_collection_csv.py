"""Adapter for BoardGameGeek collection CSV exports."""

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


class BoardGameGeekCollectionCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "boardgamegeek_collection_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["board_game", "publisher"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or ["board_game"])
        if not allowed.intersection(self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        games: list[KnowledgeUnit] = []
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
                games.append(unit)
        publishers = self._publisher_units(games) if "publisher" in allowed else []
        if "board_game" in allowed:
            result.units.extend(games)
        if "publisher" in allowed:
            result.units.extend(publishers)
        if {"board_game", "publisher"}.issubset(allowed):
            result.edges.extend(self._publisher_game_edges(publishers))
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
        title = self._first(row, "objectname", "name", "title", "Name", "Title")
        bgg_id = self._first(row, "objectid", "bgg_id", "id", "BGG ID", "game_id")
        if not title and not bgg_id:
            return None
        year = self._parse_int(self._first(row, "yearpublished", "year_published", "year", "Year Published"))
        rating = self._parse_float(self._first(row, "rating", "Rating", "my rating", "User Rating"))
        average_rating = self._parse_float(self._first(row, "average", "average_rating", "Average Rating", "avg rating"))
        plays = self._parse_int(self._first(row, "numplays", "plays", "Plays"))
        designers = self._split_people(self._first(row, "designer", "designers", "Designer", "Designers"))
        publishers = self._split_people(self._first(row, "publisher", "publishers", "Publisher", "Publishers"))
        comments = self._first(row, "comment", "comments", "privatecomment", "collection comments", "Comments")
        flags = {
            "owned": self._parse_bool(self._first(row, "owned", "own", "Owned")),
            "wishlist": self._parse_bool(self._first(row, "wishlist", "Wishlist")),
            "preordered": self._parse_bool(self._first(row, "preordered", "preorder", "Preordered")),
            "for_trade": self._parse_bool(self._first(row, "fortrade", "for_trade", "For Trade")),
            "want_to_play": self._parse_bool(self._first(row, "wanttoplay", "want_to_play", "Want To Play")),
        }
        now = datetime.now(timezone.utc)
        metadata = {
            "bgg_id": bgg_id,
            "year_published": year,
            "rating": rating,
            "average_rating": average_rating,
            "owned": flags["owned"],
            "wishlist": flags["wishlist"],
            "preordered": flags["preordered"],
            "for_trade": flags["for_trade"],
            "want_to_play": flags["want_to_play"],
            "plays": plays,
            "designers": designers,
            "publishers": publishers,
            "collection_comments": comments,
            "source_file": source_file,
            "row": dict(row),
        }
        tags = ["boardgamegeek", "board_game"] + [key for key, value in flags.items() if value]
        return KnowledgeUnit(
            source_project=SourceProject.BOARDGAMEGEEK_COLLECTION_CSV,
            source_id=self._source_id(bgg_id, title),
            source_entity_type="board_game",
            title=self._title(title or f"BGG game {bgg_id}", year),
            content=self._content(title or f"BGG game {bgg_id}", year, rating, average_rating, plays, designers, publishers, comments),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=tags,
            created_at=now,
            updated_at=now,
        )

    def _source_id(self, bgg_id: str, title: str) -> str:
        if bgg_id:
            return f"boardgamegeek_collection_csv:{bgg_id}"
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:24]
        return f"boardgamegeek_collection_csv:{digest}"

    def _publisher_units(self, games: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for game in games:
            for publisher in game.metadata.get("publishers", []) or []:
                publisher_text = str(publisher).strip()
                if not publisher_text:
                    continue
                normalized = self._normalize_publisher(publisher_text)
                grouped.setdefault(normalized, []).append(game)
                names.setdefault(normalized, publisher_text)

        now = datetime.now(timezone.utc)
        units: list[KnowledgeUnit] = []
        for normalized, publisher_games in sorted(grouped.items()):
            game_source_ids = sorted(game.source_id for game in publisher_games)
            ratings = [rating for game in publisher_games if isinstance((rating := game.metadata.get("rating")), int | float)]
            years = [year for game in publisher_games if isinstance((year := game.metadata.get("year_published")), int)]
            publisher = names[normalized]
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.BOARDGAMEGEEK_COLLECTION_CSV,
                    source_id=self._publisher_source_id(normalized),
                    source_entity_type="publisher",
                    title=publisher,
                    content=f"BoardGameGeek publisher: {publisher}\nGames: {len(publisher_games)}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "publisher": publisher,
                        "normalized_publisher": normalized,
                        "game_count": len(publisher_games),
                        "game_source_ids": game_source_ids,
                        "owned_count": sum(1 for game in publisher_games if game.metadata.get("owned") is True),
                        "played_count": sum(1 for game in publisher_games if int(game.metadata.get("plays") or 0) > 0),
                        "average_user_rating": (sum(float(rating) for rating in ratings) / len(ratings)) if ratings else None,
                        "year_range": [min(years), max(years)] if years else None,
                    },
                    tags=["boardgamegeek", "publisher", publisher],
                    created_at=min((game.created_at for game in publisher_games), default=now),
                    updated_at=max((game.updated_at for game in publisher_games), default=now),
                )
            )
        return units

    def _publisher_game_edges(self, publishers: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for publisher in publishers:
            for game_source_id in publisher.metadata.get("game_source_ids") or []:
                edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(publisher.source_id, str(game_source_id), "publisher_contains_game"),
                        from_unit_id=publisher.source_id,
                        to_unit_id=str(game_source_id),
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.BOARDGAMEGEEK_COLLECTION_CSV.value,
                            "relation_type": "publisher_contains_game",
                        },
                    )
                )
        return edges

    def _normalize_publisher(self, publisher: str) -> str:
        return " ".join(publisher.split()).casefold()

    def _publisher_source_id(self, normalized_publisher: str) -> str:
        digest = hashlib.sha256(normalized_publisher.encode("utf-8")).hexdigest()[:24]
        return f"boardgamegeek_collection_csv:publisher:{digest}"

    def _edge_id(self, from_id: str, to_id: str, relation_type: str) -> str:
        digest = hashlib.sha256("|".join([from_id, to_id, relation_type]).encode("utf-8")).hexdigest()[:24]
        return f"boardgamegeek-collection-csv-edge:{digest}"

    def _title(self, title: str, year: int | None) -> str:
        return f"{title} ({year})" if year else title

    def _content(
        self,
        title: str,
        year: int | None,
        rating: float | None,
        average_rating: float | None,
        plays: int | None,
        designers: list[str],
        publishers: list[str],
        comments: str,
    ) -> str:
        parts = [title]
        for label, value in (("Year published", year), ("Rating", rating), ("Average rating", average_rating), ("Plays", plays)):
            if value is not None:
                parts.append(f"{label}: {value}")
        if designers:
            parts.append(f"Designers: {', '.join(designers)}")
        if publishers:
            parts.append(f"Publishers: {', '.join(publishers)}")
        if comments:
            parts.append(comments)
        return "\n".join(parts)

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

    def _parse_bool(self, value: str) -> bool:
        return value.strip().casefold() in {"1", "true", "yes", "y", "on", "owned", "wishlist"}

    def _parse_int(self, value: str) -> int | None:
        if not value:
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    def _parse_float(self, value: str) -> float | None:
        if not value or value.strip().upper() == "N/A":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _split_people(self, value: str) -> list[str]:
        people: list[str] = []
        for item in re.split(r"[;,|]", value or ""):
            person = item.strip()
            if person and person not in people:
                people.append(person)
        return people

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
