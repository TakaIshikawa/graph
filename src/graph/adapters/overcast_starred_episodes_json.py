"""Adapter for Overcast starred episode JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class OvercastStarredEpisodesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "overcast_starred_episodes_json"

    @property
    def entity_types(self) -> list[str]:
        return ["starred_episode"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "starred_episode" not in set(entity_types or self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for record in records:
                unit = self._unit_from_record(record, path.name)
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
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.json") if child.is_file())

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("episodes", "starred", "items", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        podcast = self._text(record.get("podcast_title") or record.get("podcast") or record.get("show"))
        episode = self._text(record.get("episode_title") or record.get("episode") or record.get("title"))
        episode_url = self._text(record.get("episode_url") or record.get("url") or record.get("html_url"))
        audio_url = self._text(record.get("audio_url") or record.get("enclosure_url") or record.get("media_url"))
        description = self._text(record.get("description") or record.get("summary"))
        starred_at = self._parse_datetime(record.get("starred_at") or record.get("starredDate") or record.get("created_at"))
        published_at = self._parse_datetime(record.get("published_at") or record.get("pubDate") or record.get("date_published"))
        duration = self._parse_number(record.get("duration") or record.get("duration_seconds"))
        progress = self._parse_number(record.get("progress") or record.get("position") or record.get("played_seconds"))
        if not episode and not episode_url and not audio_url:
            return None
        metadata = {
            "podcast": podcast,
            "episode": episode,
            "episode_url": episode_url,
            "audio_url": audio_url,
            "description": description,
            "starred_at": starred_at.isoformat() if starred_at else self._text(record.get("starred_at")),
            "published_at": published_at.isoformat() if published_at else self._text(record.get("published_at")),
            "duration": duration,
            "progress": progress,
            "source_file": source_file,
            "record": record,
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.OVERCAST_STARRED_EPISODES_JSON,
            source_id=self._source_id(episode_url or audio_url, podcast, episode),
            source_entity_type="starred_episode",
            title=episode or episode_url or audio_url,
            content=self._content(podcast, episode, description, episode_url),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(item for item in ["overcast", "starred_episode", podcast] if item)),
            created_at=published_at or starred_at or now,
            updated_at=starred_at or published_at or now,
        )

    def _content(self, podcast: str, episode: str, description: str, url: str) -> str:
        parts = [episode, f"Podcast: {podcast}" if podcast else "", description, f"URL: {url}" if url else ""]
        return "\n".join(part for part in parts if part)

    def _source_id(self, url: str, podcast: str, episode: str) -> str:
        digest = hashlib.sha256((url or f"{podcast}|{episode}").encode("utf-8")).hexdigest()[:24]
        return f"overcast_starred_episodes_json:{digest}"

    def _parse_number(self, value: Any) -> float | int | None:
        if value in ("", None):
            return None
        try:
            number = float(str(value).strip())
            return int(number) if number.is_integer() else number
        except ValueError:
            return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        text = self._text(value)
        if not text:
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
