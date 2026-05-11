"""Adapter for Spotify Takeout streaming history exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class SpotifyStreamingHistoryAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "spotify_streaming_history"

    @property
    def entity_types(self) -> list[str]:
        return ["play", "session"]

    def __init__(self, path: str = "", session_gap_minutes: int = 30) -> None:
        self.path = path
        self.session_gap = timedelta(minutes=session_gap_minutes)

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

        sync_at = self._sync_datetime(since) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                items = self._read_items(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue

            for item in items:
                unit = self._unit_from_item(item, path.name)
                if unit is None:
                    continue
                if sync_at and unit.created_at <= sync_at:
                    continue
                units.append(unit)

        units = sorted(units, key=lambda unit: (unit.created_at, unit.source_id))
        sessions = self._session_units(units) if "session" in allowed_types else []
        if "play" in allowed_types:
            result.units.extend(units)
        if "session" in allowed_types:
            result.units.extend(sessions)
        if "play" in allowed_types and "session" in allowed_types:
            result.edges.extend(self._contains_edges(sessions))
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []

        root = Path(self.path).expanduser()
        if root.is_file() and self._is_history_file(root):
            return [root]
        if not root.is_dir():
            return []

        return sorted(
            (child for child in root.iterdir() if child.is_file() and self._is_history_file(child)),
            key=lambda child: child.name,
        )

    def _is_history_file(self, path: Path) -> bool:
        name = path.name
        return bool(
            re.fullmatch(r"StreamingHistory_music_.*\.json", name)
            or re.fullmatch(r"endsong_.*\.json", name)
        )

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("items", "history", "plays", "data"):
                nested = parsed.get(key)
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_item(self, item: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        timestamp_text = self._first(item, "ts", "endTime", "end_time", "played_at")
        played_at = self._parse_datetime(timestamp_text)
        if played_at is None:
            return None

        track_name = self._first(
            item,
            "master_metadata_track_name",
            "trackName",
            "track_name",
            "song",
        )
        artist_name = self._first(
            item,
            "master_metadata_album_artist_name",
            "artistName",
            "artist_name",
            "artist",
        )
        album_name = self._first(
            item,
            "master_metadata_album_album_name",
            "albumName",
            "album_name",
            "album",
        )
        spotify_uri = self._first(item, "spotify_track_uri", "spotify_uri", "uri")
        ms_played = self._parse_int(self._value(item, "ms_played", "msPlayed", "ms_played"))

        if not track_name and not artist_name and not spotify_uri:
            return None

        title = self._title(track_name, artist_name)
        metadata = {
            "track_name": track_name,
            "artist_name": artist_name,
            "album_name": album_name,
            "spotify_uri": spotify_uri,
            "ms_played": ms_played,
            "platform": self._first(item, "platform"),
            "country": self._first(item, "conn_country", "country"),
            "reason_start": self._first(item, "reason_start"),
            "reason_end": self._first(item, "reason_end"),
            "shuffle": self._parse_bool(self._value(item, "shuffle")),
            "skipped": self._parse_bool(self._value(item, "skipped")),
            "offline": self._parse_bool(self._value(item, "offline")),
            "source_file": source_file,
        }

        return KnowledgeUnit(
            source_project=SourceProject.SPOTIFY_STREAMING_HISTORY,
            source_id=self._source_id(played_at, track_name, artist_name, ms_played),
            source_entity_type="play",
            title=title,
            content=self._content(track_name, artist_name, album_name, ms_played),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["spotify", "music", "listening"],
            created_at=played_at,
            updated_at=played_at,
        )

    def _source_id(
        self, played_at: datetime, track_name: str, artist_name: str, ms_played: int | None
    ) -> str:
        identifier = "|".join(
            [
                played_at.isoformat(),
                track_name.strip().lower(),
                artist_name.strip().lower(),
                "" if ms_played is None else str(ms_played),
            ]
        )
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:24]
        return f"spotify_streaming_history:{digest}"

    def _title(self, track_name: str, artist_name: str) -> str:
        if track_name and artist_name:
            return f"{track_name} - {artist_name}"
        return track_name or artist_name or "Spotify play"

    def _content(
        self, track_name: str, artist_name: str, album_name: str, ms_played: int | None
    ) -> str:
        parts = []
        if track_name:
            parts.append(f"Track: {track_name}")
        if artist_name:
            parts.append(f"Artist: {artist_name}")
        if album_name:
            parts.append(f"Album: {album_name}")
        if ms_played is not None:
            parts.append(f"Milliseconds played: {ms_played}")
        return "\n".join(parts)

    def _session_units(self, plays: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        sessions: list[list[KnowledgeUnit]] = []
        for play in plays:
            if not sessions:
                sessions.append([play])
                continue
            previous = sessions[-1][-1]
            if play.created_at - previous.created_at <= self.session_gap:
                sessions[-1].append(play)
            else:
                sessions.append([play])

        return [self._session_unit(session) for session in sessions if session]

    def _session_unit(self, plays: list[KnowledgeUnit]) -> KnowledgeUnit:
        start = min(play.created_at for play in plays)
        end = max(play.created_at for play in plays)
        artists = {str(play.metadata.get("artist_name") or "") for play in plays if play.metadata.get("artist_name")}
        tracks = {str(play.metadata.get("track_name") or "") for play in plays if play.metadata.get("track_name")}
        total_ms = sum(int(play.metadata["ms_played"]) for play in plays if isinstance(play.metadata.get("ms_played"), int))
        source_files = sorted({str(play.metadata.get("source_file")) for play in plays if play.metadata.get("source_file")})
        source_id = self._session_source_id(plays)
        return KnowledgeUnit(
            source_project=SourceProject.SPOTIFY_STREAMING_HISTORY,
            source_id=source_id,
            source_entity_type="session",
            title=f"Spotify session {start.isoformat()}",
            content=f"{len(plays)} Spotify plays from {start.isoformat()} to {end.isoformat()}",
            content_type=ContentType.METADATA,
            metadata={
                "start_at": start.isoformat(),
                "end_at": end.isoformat(),
                "play_count": len(plays),
                "total_ms_played": total_ms,
                "artist_count": len(artists),
                "track_count": len(tracks),
                "source_files": source_files,
                "play_source_ids": [play.source_id for play in plays],
            },
            tags=["spotify", "music", "listening-session"],
            created_at=start,
            updated_at=end,
        )

    def _contains_edges(self, sessions: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for session in sessions:
            for play_source_id in session.metadata.get("play_source_ids") or []:
                edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(session.source_id, str(play_source_id)),
                        from_unit_id=session.source_id,
                        to_unit_id=str(play_source_id),
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.SPOTIFY_STREAMING_HISTORY.value,
                            "relation_type": "session_contains_play",
                        },
                    )
                )
        return edges

    def _session_source_id(self, plays: list[KnowledgeUnit]) -> str:
        payload = "|".join(play.source_id for play in plays)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"spotify_streaming_history:session:{digest}"

    def _edge_id(self, session_source_id: str, play_source_id: str) -> str:
        digest = hashlib.sha256("|".join((session_source_id, play_source_id, "contains")).encode("utf-8")).hexdigest()[:24]
        return f"spotify-streaming-history-contains-{digest}"

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        value = self._value(item, *keys)
        if value is None or isinstance(value, dict | list):
            return ""
        return str(value).strip()

    def _value(self, item: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in item:
                return item[key]
        return None

    def _parse_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(float(str(value).strip()))
        except ValueError:
            return None

    def _parse_bool(self, value: Any) -> bool | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "t", "yes", "y", "1"}:
            return True
        if text in {"false", "f", "no", "n", "0"}:
            return False
        return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return self._ensure_utc(value)

        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None

        return self._ensure_utc(parsed)

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        return self._ensure_utc(since.last_sync_at)
