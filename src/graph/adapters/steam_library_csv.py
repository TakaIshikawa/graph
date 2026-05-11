"""Adapter for Steam library CSV exports."""

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


class SteamLibraryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "steam_library_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["game"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "game" not in entity_types:
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
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: unit.source_id))
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
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in csv.DictReader(handle)]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        app_id = self._first(row, "app_id", "appid", "app id", "AppID", "Application ID", "steam_appid", "Game ID")
        title = self._first(row, "title", "name", "game", "Game", "Title", "Name")
        if not app_id and not title:
            return None

        playtime = self._playtime_minutes(row)
        last_played_text = self._first(row, "last_played", "last played", "Last Played", "last_played_at", "Last Played At")
        last_played = self._parse_datetime(last_played_text)
        store_url = self._first(row, "store_url", "store url", "url", "URL", "link", "Store URL")
        if not store_url and app_id:
            store_url = f"https://store.steampowered.com/app/{app_id}/"
        platform = self._first(row, "platform", "Platform") or "steam"
        tags = self._tags(row)
        now = datetime.now(timezone.utc)

        metadata = {
            "app_id": app_id,
            "playtime_minutes": playtime,
            "last_played": last_played.isoformat() if last_played else last_played_text,
            "store_url": store_url,
            "platform": platform,
            "source_file": source_file,
            "row": dict(row),
        }
        return KnowledgeUnit(
            source_project=SourceProject.STEAM_LIBRARY_CSV,
            source_id=self._source_id(app_id, title),
            source_entity_type="game",
            title=title or f"Steam app {app_id}",
            content=self._content(title or f"Steam app {app_id}", playtime, last_played, store_url, tags),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None)},
            tags=tags,
            created_at=last_played or now,
            updated_at=last_played or now,
        )

    def _source_id(self, app_id: str, title: str) -> str:
        if app_id:
            return f"steam_library_csv:{app_id}"
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:24]
        return f"steam_library_csv:{digest}"

    def _content(self, title: str, playtime: int | None, last_played: datetime | None, store_url: str, tags: list[str]) -> str:
        parts = [title]
        if playtime is not None:
            parts.append(f"Playtime: {playtime} minutes")
        if last_played is not None:
            parts.append(f"Last played: {last_played.isoformat()}")
        if store_url:
            parts.append(f"Store URL: {store_url}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

    def _tags(self, row: dict[str, Any]) -> list[str]:
        tags = ["steam", "game"]
        for key in ("tags", "Tags", "categories", "Categories", "genres", "Genres"):
            for value in re.split(r"[,;|]", self._first(row, key)):
                tag = value.strip().lower()
                if tag and tag not in tags:
                    tags.append(tag)
        return tags

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).casefold(): value for key, value in row.items()}
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.casefold())
            if value is None:
                value = compact.get(self._normalize_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _parse_playtime_minutes(self, value: str) -> int | None:
        if not value:
            return None
        text = value.strip().lower().replace(",", "")
        try:
            number = float(re.search(r"-?\d+(?:\.\d+)?", text).group(0))  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            return None
        if "hour" in text or re.search(r"\bhrs?\b", text):
            return int(round(number * 60))
        return int(round(number))

    def _playtime_minutes(self, row: dict[str, Any]) -> int | None:
        minute_value = self._first(row, "playtime_minutes", "playtime_forever", "minutes", "Minutes Played")
        if minute_value:
            return self._parse_playtime_minutes(minute_value)
        hour_value = self._first(row, "hours", "hours_played", "Hours Played", "Playtime Hours")
        if hour_value:
            text = hour_value.strip().lower()
            if re.search(r"hour|\bhrs?\b", text):
                return self._parse_playtime_minutes(hour_value)
            try:
                match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
                return int(round(float(match.group(0)) * 60)) if match else None
            except ValueError:
                return None
        return self._parse_playtime_minutes(self._first(row, "playtime", "Playtime"))

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        text = value.strip()
        for candidate in (text, f"{text}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y", "%m/%d/%Y %H:%M", "%b %d, %Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
