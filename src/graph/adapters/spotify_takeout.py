"""Adapter for Spotify Takeout streaming history exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class SpotifyTakeoutAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "spotify_takeout"

    @property
    def entity_types(self) -> list[str]:
        return ["play"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "play" not in entity_types:
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
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

        result.units.extend(sorted(units, key=lambda unit: (unit.created_at, unit.source_id)))
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
            (child for child in root.rglob("*.json") if child.is_file() and self._is_history_file(child)),
            key=lambda child: str(child.relative_to(root)),
        )

    def _is_history_file(self, path: Path) -> bool:
        name = path.name
        return bool(
            re.fullmatch(r"StreamingHistory_music_.*\.json", name)
            or re.fullmatch(r"endsong_.*\.json", name)
            or re.fullmatch(r"Streaming_History_Audio_.*\.json", name)
        )

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("items", "history", "plays", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_item(self, item: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        played_at = self._parse_datetime(self._first(item, "ts", "endTime", "end_time", "played_at"))
        if played_at is None:
            return None

        track_name = self._first(item, "master_metadata_track_name", "trackName", "track_name", "song")
        artist_name = self._first(
            item,
            "master_metadata_album_artist_name",
            "artistName",
            "artist_name",
            "artist",
        )
        album_name = self._first(item, "master_metadata_album_album_name", "albumName", "album_name", "album")
        spotify_uri = self._first(item, "spotify_track_uri", "spotify_uri", "uri")
        ms_played = self._parse_int(self._value(item, "ms_played", "msPlayed"))

        if not track_name and not artist_name and not spotify_uri:
            return None

        metadata = {
            "artist_name": artist_name,
            "track_name": track_name,
            "album_name": album_name,
            "ms_played": ms_played,
            "played_at": played_at.isoformat(),
            "platform": self._first(item, "platform"),
            "country": self._first(item, "conn_country", "country"),
            "reason_start": self._first(item, "reason_start"),
            "reason_end": self._first(item, "reason_end"),
            "shuffle": self._parse_bool(self._value(item, "shuffle")),
            "skipped": self._parse_bool(self._value(item, "skipped")),
            "offline": self._parse_bool(self._value(item, "offline")),
            "spotify_uri": spotify_uri,
            "source_file": source_file,
        }
        return KnowledgeUnit(
            source_project=SourceProject.SPOTIFY_TAKEOUT,
            source_id=self._source_id(played_at, track_name, artist_name, spotify_uri, ms_played),
            source_entity_type="play",
            title=self._title(artist_name, track_name),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["spotify", "music"],
            created_at=played_at,
            updated_at=played_at,
        )

    def _source_id(
        self,
        played_at: datetime,
        track_name: str,
        artist_name: str,
        spotify_uri: str,
        ms_played: int | None,
    ) -> str:
        identifier = "|".join(
            [
                played_at.isoformat(),
                spotify_uri.strip().lower(),
                artist_name.strip().lower(),
                track_name.strip().lower(),
                "" if ms_played is None else str(ms_played),
            ]
        )
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:24]
        return f"spotify_takeout:{digest}"

    def _title(self, artist_name: str, track_name: str) -> str:
        if artist_name and track_name:
            return f"{artist_name} - {track_name}"
        return track_name or artist_name or "Spotify play"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [f"Played at: {metadata['played_at']}"]
        if metadata["artist_name"] or metadata["track_name"]:
            parts.append(f"Track: {self._title(metadata['artist_name'], metadata['track_name'])}")
        if metadata["album_name"]:
            parts.append(f"Album: {metadata['album_name']}")
        if metadata["ms_played"] is not None:
            parts.append(f"Milliseconds played: {metadata['ms_played']}")
        if metadata["platform"]:
            parts.append(f"Platform: {metadata['platform']}")
        if metadata["country"]:
            parts.append(f"Country: {metadata['country']}")
        if metadata["reason_start"]:
            parts.append(f"Started because: {metadata['reason_start']}")
        if metadata["reason_end"]:
            parts.append(f"Ended because: {metadata['reason_end']}")
        return "\n".join(parts)

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
