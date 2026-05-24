"""Adapter for Chess.com PGN game exports."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters._personal_exports import clean_metadata, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState

_TAG_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')


class ChesscomGamesPgnAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "chesscom_games_pgn"

    @property
    def entity_types(self) -> list[str]:
        return ["game"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "game" not in entity_types:
            return result
        sync_at = since.last_sync_at.astimezone(timezone.utc) if since else None
        for path in iter_paths(self.path, {".pgn"}):
            for index, (tags, moves) in enumerate(self._games(path)):
                unit = self._unit_from_game(tags, moves, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _games(self, path: Path) -> list[tuple[dict[str, str], str]]:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return []
        chunks = re.split(r"\n\s*\n(?=\[)", text.strip())
        games: list[tuple[dict[str, str], str]] = []
        for chunk in chunks:
            tags: dict[str, str] = {}
            move_lines: list[str] = []
            for line in chunk.splitlines():
                match = _TAG_RE.match(line.strip())
                if match:
                    tags[match.group(1)] = match.group(2)
                elif line.strip():
                    move_lines.append(line.strip())
            if tags or move_lines:
                games.append((tags, "\n".join(move_lines).strip()))
        return games

    def _unit_from_game(self, tags: dict[str, str], moves: str, source_file: str, index: int) -> KnowledgeUnit:
        white = tags.get("White", "")
        black = tags.get("Black", "")
        result = tags.get("Result", "")
        played_at = self._date(tags)
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "players": {"white": white, "black": black},
                "white": white,
                "black": black,
                "white_rating": tags.get("WhiteElo"),
                "black_rating": tags.get("BlackElo"),
                "result": result,
                "time_control": tags.get("TimeControl"),
                "opening": tags.get("Opening"),
                "eco": tags.get("ECO"),
                "date": played_at.isoformat() if played_at else tags.get("Date"),
                "termination": tags.get("Termination"),
                "url": tags.get("Link") or tags.get("Site"),
                "source_url": tags.get("Link") or tags.get("Site"),
                "move_text": moves,
                "pgn_tags": dict(tags),
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="chesscom_games_pgn",
            source_id=self._source_id(tags, moves, index),
            source_entity_type="game",
            title=f"{white or 'White'} vs {black or 'Black'} {result}".strip(),
            content=self._content(tags, moves),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["chess", "chess.com"],
            created_at=played_at or now,
            updated_at=played_at or now,
        )

    def _date(self, tags: dict[str, str]) -> datetime | None:
        date = tags.get("UTCDate") or tags.get("Date") or ""
        time = tags.get("UTCTime") or ""
        if time:
            return parse_datetime(f"{date.replace('.', '-')} {time}")
        return parse_datetime(date.replace(".", "-"))

    def _source_id(self, tags: dict[str, str], moves: str, index: int) -> str:
        raw = tags.get("Link") or "|".join([tags.get("UTCDate", ""), tags.get("White", ""), tags.get("Black", ""), moves, str(index)])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"chesscom_games_pgn:{digest}"

    def _content(self, tags: dict[str, str], moves: str) -> str:
        parts = [f"{tags.get('White', 'White')} vs {tags.get('Black', 'Black')}"]
        for key in ("Result", "TimeControl", "Opening", "ECO", "Termination", "Link"):
            if tags.get(key):
                parts.append(f"{key}: {tags[key]}")
        if moves:
            parts.append(moves)
        return "\n".join(parts)
