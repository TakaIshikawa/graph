"""Adapter for Apple Music library CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class AppleMusicLibraryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "apple_music_library_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["song"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if "song" not in allowed_types:
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
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
                result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
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
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = self._first(row, "Title", "Name", "Song", "Track", "Track Name")
        artist = self._first(row, "Artist", "Artist Name", "Track Artist")
        album = self._first(row, "Album", "Album Name")
        genre = self._first(row, "Genre", "Genres")
        play_count = self._parse_int(self._first(row, "Play Count", "Plays", "Played Count", "Playcount"))
        rating = self._parse_int(self._first(row, "Rating", "My Rating", "Stars"))
        last_played_text = self._first(row, "Last Played", "Last Played Date", "Last Played At", "Played At", "Last Play")
        last_played = self._parse_datetime(last_played_text)
        date_added_text = self._first(row, "Date Added", "Added Date")
        date_added = self._parse_datetime(date_added_text)
        last_skipped_text = self._first(row, "Last Skipped", "Last Skipped Date")
        last_skipped = self._parse_datetime(last_skipped_text)
        persistent_id = self._first(row, "Persistent ID", "Track ID", "ID", "Apple Music ID")
        if not title and not persistent_id:
            return None
        metadata = {
            "title": title,
            "artist": artist,
            "album": album,
            "genre": genre,
            "play_count": play_count,
            "rating": rating,
            "last_played": last_played.isoformat() if last_played else last_played_text,
            "date_added": date_added.isoformat() if date_added else date_added_text,
            "skip_count": self._parse_int(self._first(row, "Skip Count", "Skips")),
            "last_skipped": last_skipped.isoformat() if last_skipped else last_skipped_text,
            "loved": self._parse_bool(self._first(row, "Loved", "Favorite")),
            "cloud_status": self._first(row, "Cloud Status"),
            "persistent_id": persistent_id,
            "source_file": source_file,
            "row": dict(row),
        }
        now = datetime.now(timezone.utc)
        tags = ["apple_music", "song", genre]
        return KnowledgeUnit(
            source_project=SourceProject.APPLE_MUSIC_LIBRARY_CSV,
            source_id=self._source_id(persistent_id, title, artist, album),
            source_entity_type="song",
            title=title or persistent_id,
            content=self._content(title, artist, album, metadata),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(item for item in tags if item)),
            created_at=last_played or now,
            updated_at=last_played or now,
        )

    def _content(self, title: str, artist: str, album: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (
            ("artist", "Artist"),
            ("album", "Album"),
            ("genre", "Genre"),
            ("play_count", "Play count"),
            ("rating", "Rating"),
            ("last_played", "Last played"),
            ("date_added", "Date added"),
            ("skip_count", "Skip count"),
            ("last_skipped", "Last skipped"),
            ("loved", "Loved"),
            ("cloud_status", "Cloud status"),
        ):
            if metadata.get(key) not in ("", None):
                parts.append(f"{label}: {metadata[key]}")
        if artist and album:
            return "\n".join(item for item in parts if item)
        return "\n".join(item for item in parts if item)

    def _source_id(self, persistent_id: str, title: str, artist: str, album: str) -> str:
        raw = persistent_id or "|".join([self._normalized(title), self._normalized(artist), self._normalized(album)])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"apple_music_library_csv:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_int(self, value: str) -> int | None:
        if not value:
            return None
        try:
            return int(float(value.strip()))
        except ValueError:
            return None

    def _parse_bool(self, value: str) -> bool | None:
        text = value.strip().casefold()
        if not text:
            return None
        if text in {"1", "true", "yes", "y", "loved", "favorite", "favorited"}:
            return True
        if text in {"0", "false", "no", "n", "unloved", "not loved"}:
            return False
        return None

    def _parse_datetime(self, value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        for candidate in (text, text.replace("Z", "+00:00")):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate))
            except ValueError:
                pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _normalized(self, value: str) -> str:
        return " ".join(value.casefold().split())

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
