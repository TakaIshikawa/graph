"""Adapter for Last.fm scrobbles CSV exports."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class LastfmScrobblesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "lastfm_scrobbles_csv"

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
        allowed_types = set(entity_types or self.entity_types)
        if "play" not in allowed_types:
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

        units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        self._add_replay_metadata(units)
        result.units.extend(units)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.csv") if path.is_file())

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [
                {str(key).strip(): value for key, value in row.items() if key is not None}
                for row in csv.DictReader(handle)
            ]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        listened_at = self._parse_datetime(
            self._first(
                row,
                "listened_at",
                "scrobbled_at",
                "timestamp",
                "date",
                "datetime",
                "time",
                "uts",
                "utc_time",
            )
        )
        artist = self._first(row, "artist", "artist_name", "Artist", "Artist Name")
        album = self._first(row, "album", "album_name", "Album", "Album Name")
        track = self._first(row, "track", "track_name", "title", "Title", "song", "name")
        if listened_at is None or not (artist or track):
            return None

        album_artist = self._first(row, "album_artist", "album artist", "albumArtist", "Album Artist")
        duration_text = self._first(row, "duration", "duration_seconds", "duration_ms", "length", "track_duration")
        duration_seconds = self._parse_duration_seconds(duration_text)
        track_url = self._first(row, "track_url", "url", "URL", "lastfm_url", "link")
        loved = self._parse_bool(self._first(row, "loved", "love", "Loved", "is_loved"))
        skipped = self._parse_bool(self._first(row, "skipped", "skip", "Skipped", "is_skipped"))
        metadata = {
            "artist": artist,
            "album": album,
            "track": track,
            "listened_at": listened_at.isoformat(),
            "duration": duration_text,
            "duration_seconds": duration_seconds,
            "track_url": track_url,
            "album_artist": album_artist,
            "loved": loved,
            "skipped": skipped,
            "source_file": source_file,
            "row": dict(row),
        }
        return KnowledgeUnit(
            source_project=SourceProject.LASTFM_SCROBBLES_CSV,
            source_id=self._source_id(listened_at, artist, album, track),
            source_entity_type="play",
            title=self._title(track, artist),
            content=self._content(track, artist, album, listened_at, duration_seconds, track_url),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["lastfm", "music", "listening"],
            created_at=listened_at,
            updated_at=listened_at,
        )

    def _source_id(self, listened_at: datetime, artist: str, album: str, track: str) -> str:
        raw = "|".join(
            [
                listened_at.isoformat(),
                artist.strip().casefold(),
                album.strip().casefold(),
                track.strip().casefold(),
            ]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"lastfm_scrobbles_csv:play:{digest}"

    def _add_replay_metadata(self, units: list[KnowledgeUnit]) -> None:
        previous_by_identity: dict[tuple[str, str, str], datetime] = {}
        sequence_by_identity: dict[tuple[str, str, str], int] = {}
        for unit in units:
            identity = self._play_identity(unit.metadata)
            sequence = sequence_by_identity.get(identity, 0) + 1
            sequence_by_identity[identity] = sequence
            unit.metadata["play_sequence"] = sequence
            unit.metadata["is_replay"] = sequence > 1
            previous_played_at = previous_by_identity.get(identity)
            if previous_played_at is not None:
                unit.metadata["previous_played_at"] = previous_played_at.isoformat()
            previous_by_identity[identity] = unit.created_at

    def _play_identity(self, metadata: dict[str, Any]) -> tuple[str, str, str]:
        return (
            self._normalize_text(metadata.get("artist")),
            self._normalize_text(metadata.get("album")),
            self._normalize_text(metadata.get("track")),
        )

    def _title(self, track: str, artist: str) -> str:
        if track and artist:
            return f"{track} - {artist}"
        return track or artist or "Last.fm scrobble"

    def _content(
        self,
        track: str,
        artist: str,
        album: str,
        listened_at: datetime,
        duration_seconds: int | None,
        track_url: str,
    ) -> str:
        parts = []
        if track:
            parts.append(f"Track: {track}")
        if artist:
            parts.append(f"Artist: {artist}")
        if album:
            parts.append(f"Album: {album}")
        parts.append(f"Listened at: {listened_at.isoformat()}")
        if duration_seconds is not None:
            parts.append(f"Duration seconds: {duration_seconds}")
        if track_url:
            parts.append(f"URL: {track_url}")
        return "\n".join(parts)

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).strip().casefold(): value for key, value in row.items()}
        compact = {self._compact_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.casefold())
            if value is None:
                value = compact.get(self._compact_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _compact_key(self, value: Any) -> str:
        return "".join(ch for ch in str(value).casefold() if ch.isalnum())

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        text = str(value).strip()
        if text.isdigit():
            try:
                return datetime.fromtimestamp(int(text), tz=timezone.utc)
            except (OverflowError, ValueError, OSError):
                return None
        for candidate in (text, f"{text}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                pass
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
            "%d %b %Y, %H:%M",
        ):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _parse_duration_seconds(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        text = str(value).strip()
        if ":" in text:
            parts = text.split(":")
            try:
                total = 0
                for part in parts:
                    total = total * 60 + int(float(part))
                return total
            except ValueError:
                return None
        try:
            parsed = float(text)
        except ValueError:
            return None
        if "ms" in text.casefold() or parsed > 10_000:
            return int(parsed / 1000)
        return int(parsed)

    def _parse_bool(self, value: Any) -> bool | None:
        if value is None or value == "":
            return None
        text = str(value).strip().casefold()
        if text in {"1", "true", "t", "yes", "y", "loved"}:
            return True
        if text in {"0", "false", "f", "no", "n", "none"}:
            return False
        return None

    def _normalize_text(self, value: Any) -> str:
        return " ".join(str(value or "").strip().casefold().split())

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
