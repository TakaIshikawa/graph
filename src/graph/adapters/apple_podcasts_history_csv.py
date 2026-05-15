"""Adapter for Apple Podcasts listening history CSV exports."""

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


class ApplePodcastsHistoryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "apple_podcasts_history_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["podcast_episode"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "podcast_episode" not in set(entity_types or self.entity_types):
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
        show = self._first(row, "Show", "Podcast", "Podcast Title", "Show Name")
        episode = self._first(row, "Episode", "Episode Title", "Title", "Name")
        played_at = self._parse_datetime(self._first(row, "Played At", "Played Date", "Date Played", "Listen Date", "Last Played"))
        url = self._first(row, "URL", "Episode URL", "Link", "Guid")
        duration = self._parse_duration(self._first(row, "Duration", "Duration Seconds", "Length"))
        progress = self._parse_duration(self._first(row, "Progress", "Progress Seconds", "Position", "Elapsed"))
        completed = self._parse_bool(self._first(row, "Completed", "Complete", "Finished", "Played"))
        if not episode and not show and not url:
            return None
        metadata = {
            "show": show,
            "episode": episode,
            "duration": duration,
            "progress": progress,
            "completed": completed,
            "played_at": played_at.isoformat() if played_at else self._first(row, "Played At", "Played Date", "Date Played", "Listen Date", "Last Played"),
            "url": url,
            "source_file": source_file,
            "row": dict(row),
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.APPLE_PODCASTS_HISTORY_CSV,
            source_id=self._source_id(show, episode, played_at, url),
            source_entity_type="podcast_episode",
            title=episode or show or url,
            content=self._content(show, episode, metadata),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(item for item in ["apple_podcasts", "podcast_episode", show] if item)),
            created_at=played_at or now,
            updated_at=played_at or now,
        )

    def _content(self, show: str, episode: str, metadata: dict[str, Any]) -> str:
        parts = [episode]
        for key, label in (("show", "Show"), ("duration", "Duration"), ("progress", "Progress"), ("completed", "Completed"), ("played_at", "Played"), ("url", "URL")):
            if metadata.get(key) not in ("", None):
                parts.append(f"{label}: {metadata[key]}")
        if show and not episode:
            parts.insert(0, show)
        return "\n".join(str(item) for item in parts if item)

    def _source_id(self, show: str, episode: str, played_at: datetime | None, url: str) -> str:
        raw = url or "|".join([self._normalized(show), self._normalized(episode), played_at.isoformat() if played_at else ""])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"apple_podcasts_history_csv:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

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
        if text in {"true", "yes", "y", "1", "complete", "completed", "finished"}:
            return True
        if text in {"false", "no", "n", "0", "incomplete", "unfinished"}:
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
