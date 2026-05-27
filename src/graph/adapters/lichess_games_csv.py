"""Adapter for Lichess game CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class LichessGamesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "lichess_games_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["chess_game"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "chess_game" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        game_id = first(row, "Game ID", "ID", "GameId")
        url = first(row, "URL", "Link")
        white = first(row, "White", "White Player")
        black = first(row, "Black", "Black Player")
        result = first(row, "Result", "Outcome")
        if not any((game_id, url, white, black, result)):
            return None
        played_at = parse_datetime(first(row, "Date", "UTC Date", "Played At", "Created At")) or datetime.now(timezone.utc)
        opening = first(row, "Opening")
        eco = first(row, "ECO")
        time_control = first(row, "Time Control", "TimeControl")
        rated = self._bool(first(row, "Rated", "Is Rated"))
        moves = first(row, "Moves", "Move Text", "PGN")
        metadata = clean_metadata(
            {
                "game_id": game_id,
                "url": url,
                "white": white,
                "black": black,
                "result": result,
                "opening": opening,
                "eco": eco,
                "time_control": time_control,
                "rated": rated,
                "moves": moves,
                "date": played_at.isoformat(),
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="lichess_games_csv",
            source_id=self._source_id(game_id, url, white, black, result, played_at, moves, index),
            source_entity_type="chess_game",
            title=f"{white or 'White'} vs {black or 'Black'} {result}".strip(),
            content=self._content(white, black, result, opening, eco, time_control, url, moves),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["chess", "lichess"],
            created_at=played_at,
            updated_at=played_at,
        )

    def _source_id(self, game_id: str, url: str, white: str, black: str, result: str, played_at: datetime, moves: str, index: int) -> str:
        if game_id:
            return f"lichess_games_csv:{game_id}"
        if url:
            return digest_source_id("lichess_games_csv", url)
        return digest_source_id("lichess_games_csv", played_at.isoformat(), white, black, result, moves, index)

    def _content(self, white: str, black: str, result: str, opening: str, eco: str, time_control: str, url: str, moves: str) -> str:
        parts = [f"{white or 'White'} vs {black or 'Black'}"]
        for label, value in (("Result", result), ("Opening", opening), ("ECO", eco), ("Time control", time_control), ("URL", url)):
            if value:
                parts.append(f"{label}: {value}")
        if moves:
            parts.append(moves)
        return "\n".join(parts)

    def _bool(self, value: str) -> bool | None:
        if not value:
            return None
        normalized = value.casefold()
        if normalized in {"1", "true", "yes", "y", "rated"}:
            return True
        if normalized in {"0", "false", "no", "n", "casual", "unrated"}:
            return False
        return None
