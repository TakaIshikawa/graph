"""Adapter for Pocket Casts listening history CSV exports."""

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


class PocketCastsListeningHistoryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pocket_casts_listening_history_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["podcast_listen"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "podcast_listen" not in set(entity_types or self.entity_types):
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
        podcast = self._first(row, "podcast", "podcast_title", "Podcast", "Podcast Title", "Show", "Show Title")
        episode = self._first(row, "episode", "episode_title", "Episode", "Episode Title", "Title")
        url = self._first(row, "url", "episode_url", "URL", "Episode URL", "Link")
        played_at = self._parse_datetime(self._first(row, "played_at", "Played At", "Last Played", "Date Played"))
        completed_at = self._parse_datetime(self._first(row, "completed_at", "Completed At", "Finished At"))
        duration = self._parse_duration(self._first(row, "duration", "Duration", "Duration Seconds", "Length"))
        progress = self._parse_number_or_duration(self._first(row, "progress", "Progress", "Position", "Played Seconds"))
        archived = self._parse_bool(self._first(row, "archived", "Archived", "Is Archived"))
        favorite = self._parse_bool(self._first(row, "favorite", "favourite", "Favorite", "Starred"))
        if not podcast and not episode and not url:
            return None
        updated_at = completed_at or played_at
        metadata = {
            "podcast": podcast,
            "episode": episode,
            "url": url,
            "duration": duration,
            "progress": progress,
            "played_at": played_at.isoformat() if played_at else self._first(row, "played_at", "Played At", "Last Played", "Date Played"),
            "completed_at": completed_at.isoformat() if completed_at else self._first(row, "completed_at", "Completed At", "Finished At"),
            "archived": archived,
            "favorite": favorite,
            "source_file": source_file,
            "row": dict(row),
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.POCKET_CASTS_LISTENING_HISTORY_CSV,
            source_id=self._source_id(podcast, episode, updated_at, url),
            source_entity_type="podcast_listen",
            title=episode or podcast or url,
            content=self._content(podcast, episode, metadata),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(item for item in ["pocket_casts", "podcast", "listen", podcast] if item)),
            created_at=played_at or completed_at or now,
            updated_at=updated_at or now,
        )

    def _content(self, podcast: str, episode: str, metadata: dict[str, Any]) -> str:
        parts = [episode or podcast]
        for key, label in (("podcast", "Podcast"), ("duration", "Duration"), ("progress", "Progress"), ("played_at", "Played"), ("completed_at", "Completed"), ("url", "URL")):
            if metadata.get(key) not in ("", None):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(str(part) for part in parts if part)

    def _source_id(self, podcast: str, episode: str, updated_at: datetime | None, url: str) -> str:
        raw = url or "|".join([self._normalized(podcast), self._normalized(episode), updated_at.isoformat() if updated_at else ""])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"pocket_casts_listening_history_csv:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_number_or_duration(self, value: str) -> float | int | None:
        return self._parse_duration(value)

    def _parse_duration(self, value: str) -> float | int | None:
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
            return int(number) if number.is_integer() else number
        except ValueError:
            pass
        parts = [part for part in text.split(":") if part]
        if len(parts) in {2, 3} and all(part.isdigit() for part in parts):
            seconds = 0
            for part in parts:
                seconds = seconds * 60 + int(part)
            return seconds
        return None

    def _parse_bool(self, value: str) -> bool | None:
        text = value.strip().casefold()
        if not text:
            return None
        if text in {"true", "yes", "y", "1", "favorite", "favourite", "starred", "archived"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False
        return None

    def _parse_datetime(self, value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
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
