"""Adapter for Spotify saved podcast episodes CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SpotifySavedEpisodesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "spotify_saved_episodes_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["podcast_episode"]

    def __init__(self, path: str | Path | TextIO = "", file: TextIO | None = None) -> None:
        self.path = path
        self.file = file

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "podcast_episode" not in set(entity_types or self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for rows, source_file in self._iter_row_sets():
            for row in rows:
                unit = self._unit_from_row(row, source_file)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _iter_row_sets(self) -> list[tuple[list[dict[str, str]], str]]:
        if self.file is not None:
            return [(self._read_rows_from_handle(self.file), self._source_name(self.file))]
        if hasattr(self.path, "read"):
            handle = self.path
            return [(self._read_rows_from_handle(handle), self._source_name(handle))]
        if not self.path:
            return []

        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            paths = [root]
        elif root.is_dir():
            paths = sorted(child for child in root.rglob("*.csv") if child.is_file())
        else:
            return []

        row_sets: list[tuple[list[dict[str, str]], str]] = []
        for path in paths:
            try:
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    row_sets.append((self._read_rows_from_handle(handle), path.name))
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
        return row_sets

    def _read_rows_from_handle(self, handle: Any) -> list[dict[str, str]]:
        if hasattr(handle, "seek"):
            handle.seek(0)
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        return [{str(key).strip(): "" if value is None else str(value).strip() for key, value in row.items() if key is not None} for row in reader]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        if not any(self._text(value) for value in row.values()):
            return None

        show = self._first(row, "Show Name", "Show", "Podcast", "Podcast Title", "show_name")
        episode = self._first(row, "Episode Name", "Episode", "Episode Title", "Title", "episode_name")
        uri = self._first(row, "Episode URI", "Spotify URI", "URI", "episode_uri")
        url = self._first(row, "Episode URL", "URL", "Link", "episode_url")
        added_at_text = self._first(row, "Added At", "Added Date", "Saved At", "Date Added", "added_at")
        release_date_text = self._first(row, "Release Date", "Released", "Published At", "release_date")
        added_at = self._parse_datetime(added_at_text)
        release_date = self._parse_datetime(release_date_text)
        duration_ms = self._parse_int(self._first(row, "Duration Ms", "Duration MS", "Duration Milliseconds", "duration_ms"))
        description = self._first(row, "Description", "Episode Description", "description")
        explicit = self._parse_bool(self._first(row, "Explicit", "Is Explicit", "explicit"))

        if not show and not episode and not uri and not url:
            return None

        metadata = {
            "show": show,
            "episode": episode,
            "episode_uri": uri,
            "episode_url": url,
            "added_at": added_at.isoformat() if added_at else added_at_text,
            "release_date": release_date.isoformat() if release_date else release_date_text,
            "duration_ms": duration_ms,
            "description": description,
            "explicit": explicit,
            "source_file": source_file,
            "row": dict(row),
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project="spotify_saved_episodes_csv",
            source_id=self._source_id(show, episode, uri, url, added_at, release_date),
            source_entity_type="podcast_episode",
            title=episode or show or uri or url,
            content=self._content(show, episode, metadata),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=self._dedupe(["spotify", "podcast", "podcast_episode", show]),
            created_at=added_at or release_date or now,
            updated_at=added_at or release_date or now,
        )

    def _content(self, show: str, episode: str, metadata: dict[str, Any]) -> str:
        parts = []
        if episode:
            parts.append(f"Episode: {episode}")
        if show:
            parts.append(f"Show: {show}")
        for key, label in (
            ("episode_uri", "Spotify URI"),
            ("episode_url", "URL"),
            ("added_at", "Added"),
            ("release_date", "Released"),
            ("duration_ms", "Duration ms"),
            ("explicit", "Explicit"),
        ):
            if metadata.get(key) not in ("", None):
                parts.append(f"{label}: {metadata[key]}")
        if metadata.get("description"):
            parts.append(f"\nDescription:\n{metadata['description']}")
        return "\n".join(str(part) for part in parts if part)

    def _source_id(self, show: str, episode: str, uri: str, url: str, added_at: datetime | None, release_date: datetime | None) -> str:
        raw = uri or url or "|".join(
            [
                self._stable_text(show),
                self._stable_text(episode),
                added_at.isoformat() if added_at else "",
                release_date.isoformat() if release_date else "",
            ]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"spotify_saved_episodes_csv:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            text = self._text(value)
            if text:
                return text
        return ""

    def _parse_int(self, value: str) -> int | None:
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None

    def _parse_bool(self, value: str) -> bool | None:
        text = value.strip().casefold()
        if not text:
            return None
        if text in {"true", "yes", "y", "1", "explicit"}:
            return True
        if text in {"false", "no", "n", "0", "clean"}:
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

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _stable_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().casefold())

    def _text(self, value: Any) -> str:
        if value is None or isinstance(value, (dict, list)):
            return ""
        return str(value).strip()

    def _dedupe(self, values: Any) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    def _source_name(self, handle: Any) -> str:
        name = self._text(getattr(handle, "name", ""))
        return Path(name).name if name else "<memory>"
