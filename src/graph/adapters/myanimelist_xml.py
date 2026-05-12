"""Adapter for MyAnimeList XML exports."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class MyAnimeListXmlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "myanimelist_xml"

    @property
    def entity_types(self) -> list[str]:
        return ["anime"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "anime" not in entity_types:
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                root = ET.parse(path).getroot()
            except (OSError, UnicodeDecodeError, ET.ParseError):
                continue
            for element in root.findall(".//anime"):
                try:
                    unit = self._unit_from_element(element, path)
                except (TypeError, ValueError):
                    continue
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _iter_paths(self) -> list[Path]:
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".xml":
            return [root]
        if root.is_dir():
            return sorted(child for child in root.rglob("*.xml") if child.is_file())
        return []

    def _unit_from_element(self, element: ET.Element, path: Path) -> KnowledgeUnit | None:
        fields = {child.tag.strip(): (child.text or "").strip() for child in list(element)}
        title = self._first(fields, "series_title", "anime_title", "title")
        if not title:
            return None

        mal_id = self._first(fields, "series_animedb_id", "mal_id", "id")
        anime_type = self._first(fields, "series_type", "type")
        status = self._first(fields, "my_status", "status")
        watched = self._parse_int(self._first(fields, "my_watched_episodes", "watched_episodes"))
        total = self._parse_int(self._first(fields, "series_episodes", "total_episodes"))
        score = self._parse_int(self._first(fields, "my_score", "score"))
        tags = self._split_tags(self._first(fields, "my_tags", "tags"))
        start_date = self._parse_date(self._first(fields, "my_start_date", "start_date"))
        end_date = self._parse_date(self._first(fields, "my_finish_date", "my_end_date", "end_date"))
        storage = self._first(fields, "my_storage", "storage")
        rewatching = self._bool_or_none(self._first(fields, "my_rewatching", "rewatching"))
        rewatch_count = self._parse_int(self._first(fields, "my_times_watched", "times_watched"))
        updated_at = end_date or start_date or datetime.now(timezone.utc)

        metadata = {
            "mal_id": mal_id,
            "title": title,
            "type": anime_type,
            "status": status,
            "watched_episodes": watched,
            "total_episodes": total,
            "score": score,
            "tags": tags,
            "start_date": start_date.isoformat() if start_date else "",
            "end_date": end_date.isoformat() if end_date else "",
            "storage": storage,
            "rewatching": rewatching,
            "rewatch_count": rewatch_count,
            "source_file": str(path),
            "fields": fields,
        }
        return KnowledgeUnit(
            source_project=SourceProject.MYANIMELIST_XML,
            source_id=self._source_id(mal_id, fields),
            source_entity_type="anime",
            title=title,
            content=self._content(title, anime_type, status, watched, total, score, tags),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=self._dedupe(["myanimelist", *tags, status.lower() if status else ""]),
            created_at=start_date or updated_at,
            updated_at=updated_at,
        )

    def _source_id(self, mal_id: str, fields: dict[str, str]) -> str:
        if mal_id:
            return f"myanimelist_xml:{mal_id}"
        raw = "|".join(f"{key}={fields[key]}" for key in sorted(fields))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"myanimelist_xml:{digest}"

    def _content(
        self,
        title: str,
        anime_type: str,
        status: str,
        watched: int | None,
        total: int | None,
        score: int | None,
        tags: list[str],
    ) -> str:
        parts = [f"Title: {title}"]
        if anime_type:
            parts.append(f"Type: {anime_type}")
        if status:
            parts.append(f"Status: {status}")
        if watched is not None:
            progress = str(watched) if total is None else f"{watched}/{total}"
            parts.append(f"Episodes: {progress}")
        if score is not None:
            parts.append(f"Score: {score}/10")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

    def _first(self, fields: dict[str, str], *keys: str) -> str:
        lowered = {key.lower(): value for key, value in fields.items()}
        for key in keys:
            value = fields.get(key) or lowered.get(key.lower())
            if value:
                return value.strip()
        return ""

    def _split_tags(self, value: str) -> list[str]:
        return self._dedupe(part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip())

    def _dedupe(self, values: Any) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in result:
                result.append(text)
        return result

    def _parse_int(self, value: str) -> int | None:
        if not value:
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    def _bool_or_none(self, value: str) -> bool | None:
        if not value:
            return None
        return value.strip().lower() in {"1", "true", "yes"}

    def _parse_date(self, value: str) -> datetime | None:
        if not value or value in {"0000-00-00", "0000-00-00 00:00:00"}:
            return None
        for candidate in (value, f"{value}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
