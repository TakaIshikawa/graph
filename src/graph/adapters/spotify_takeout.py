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
        return ["play", "podcast_play"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or ["play"])
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                items = self._read_items(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for item in items:
                unit = self._unit_from_item(item, path.name, allowed_types)
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

    def _unit_from_item(
        self,
        item: dict[str, Any],
        source_file: str,
        allowed_types: set[str],
    ) -> KnowledgeUnit | None:
        played_at = self._parse_datetime(self._first(item, "ts", "endTime", "end_time", "played_at"))
        if played_at is None:
            return None

        if "podcast_play" in allowed_types and self._is_podcast_item(item):
            return self._podcast_unit_from_item(item, source_file, played_at)
        if "play" not in allowed_types:
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

    def _podcast_unit_from_item(
        self,
        item: dict[str, Any],
        source_file: str,
        played_at: datetime,
    ) -> KnowledgeUnit | None:
        episode_name = self._first(
            item,
            "episode_name",
            "episodeName",
            "master_metadata_episode_name",
            "podcast_episode_name",
        )
        show_name = self._first(
            item,
            "episode_show_name",
            "episodeShowName",
            "master_metadata_show_name",
            "podcast_show_name",
            "show_name",
        )
        spotify_uri = self._first(item, "spotify_episode_uri", "spotifyEpisodeUri", "episode_uri", "spotify_uri", "uri")
        ms_played = self._parse_int(self._value(item, "ms_played", "msPlayed"))

        if not episode_name and not show_name and not spotify_uri:
            return None

        metadata = {
            "show_name": show_name,
            "episode_name": episode_name,
            "spotify_uri": spotify_uri,
            "ms_played": ms_played,
            "played_at": played_at.isoformat(),
            "platform": self._first(item, "platform"),
            "country": self._first(item, "conn_country", "country"),
            "reason_start": self._first(item, "reason_start"),
            "reason_end": self._first(item, "reason_end"),
            "skipped": self._parse_bool(self._value(item, "skipped")),
            "offline": self._parse_bool(self._value(item, "offline")),
            "incognito_mode": self._parse_bool(self._value(item, "incognito_mode", "incognitoMode")),
            "source_file": source_file,
        }
        return KnowledgeUnit(
            source_project=SourceProject.SPOTIFY_TAKEOUT,
            source_id=self._podcast_source_id(played_at, show_name, episode_name, spotify_uri, ms_played),
            source_entity_type="podcast_play",
            title=self._podcast_title(show_name, episode_name),
            content=self._podcast_content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["spotify", "podcast"],
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

    def _podcast_source_id(
        self,
        played_at: datetime,
        show_name: str,
        episode_name: str,
        spotify_uri: str,
        ms_played: int | None,
    ) -> str:
        identifier = "|".join(
            [
                "podcast",
                played_at.isoformat(),
                spotify_uri.strip().lower(),
                show_name.strip().lower(),
                episode_name.strip().lower(),
                "" if ms_played is None else str(ms_played),
            ]
        )
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:24]
        return f"spotify_takeout:podcast_play:{digest}"

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

    def _podcast_title(self, show_name: str, episode_name: str) -> str:
        if show_name and episode_name:
            return f"{show_name} - {episode_name}"
        return episode_name or show_name or "Spotify podcast play"

    def _podcast_content(self, metadata: dict[str, Any]) -> str:
        parts = [f"Played at: {metadata['played_at']}"]
        if metadata["show_name"] or metadata["episode_name"]:
            parts.append(f"Episode: {self._podcast_title(metadata['show_name'], metadata['episode_name'])}")
        if metadata["spotify_uri"]:
            parts.append(f"URI: {metadata['spotify_uri']}")
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

    def _is_podcast_item(self, item: dict[str, Any]) -> bool:
        return bool(
            self._first(
                item,
                "episode_name",
                "episodeName",
                "master_metadata_episode_name",
                "podcast_episode_name",
                "episode_show_name",
                "episodeShowName",
                "master_metadata_show_name",
                "podcast_show_name",
                "show_name",
                "spotify_episode_uri",
                "spotifyEpisodeUri",
                "episode_uri",
            )
        )

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
