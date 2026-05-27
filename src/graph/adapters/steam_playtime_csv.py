"""Adapter for manually exported Steam playtime CSV files."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SteamPlaytimeCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "steam_playtime_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["game_playtime"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "game_playtime" not in set(entity_types or self.entity_types):
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

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if root.is_dir():
            return sorted(path for path in root.rglob("*.csv") if path.is_file())
        return []

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in csv.DictReader(handle)]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        app_id = self._first(row, "app_id", "appid", "app id", "AppID", "Application ID", "steam_appid", "Game ID")
        game_name = self._first(row, "game", "game name", "name", "title", "Game", "Name", "Title")
        store_url = self._first(row, "store_url", "store url", "url", "URL", "link", "Store URL")
        if not app_id and not game_name and not store_url:
            return None

        hours_played = self._parse_hours(self._first(row, "hours_played", "Hours Played", "hours", "Hours", "playtime hours"))
        minutes_played = self._parse_minutes(self._first(row, "minutes_played", "Minutes Played", "playtime_forever"))
        if hours_played is None and minutes_played is not None:
            hours_played = round(minutes_played / 60, 2)
        last_played_text = self._first(row, "last_played", "Last Played", "last played", "last_played_at")
        first_played_text = self._first(row, "first_played", "First Played", "first played", "first_played_at")
        last_played = self._parse_datetime(last_played_text)
        first_played = self._parse_datetime(first_played_text)
        achievements_unlocked = self._parse_int(self._first(row, "achievements_unlocked", "Achievements Unlocked", "unlocked"))
        achievements_total = self._parse_int(self._first(row, "achievements_total", "Achievements Total", "total achievements", "total"))
        platform = self._first(row, "platform", "Platform")
        platforms = self._split_values(self._first(row, "platforms", "Platforms", "owned_platforms", "Owned Platforms") or platform)
        if not store_url and app_id:
            store_url = f"https://store.steampowered.com/app/{app_id}/"

        now = datetime.now(timezone.utc)
        title = game_name or f"Steam app {app_id}"
        metadata = {
            "app_id": app_id,
            "game_name": game_name,
            "hours_played": hours_played,
            "minutes_played": minutes_played,
            "last_played": last_played.isoformat() if last_played else last_played_text,
            "first_played": first_played.isoformat() if first_played else first_played_text,
            "achievements_unlocked": achievements_unlocked,
            "achievements_total": achievements_total,
            "store_url": store_url,
            "platform": platform,
            "platforms": platforms,
            "source_file": source_file,
            "row": dict(row),
        }
        return KnowledgeUnit(
            source_project=self.name,
            source_id=self._source_id(app_id, game_name or store_url),
            source_entity_type="game_playtime",
            title=title,
            content=self._content(title, hours_played, last_played, achievements_unlocked, achievements_total, store_url),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=["steam", "game_playtime", *platforms],
            created_at=first_played or last_played or now,
            updated_at=last_played or first_played or now,
        )

    def _content(
        self,
        title: str,
        hours_played: float | None,
        last_played: datetime | None,
        achievements_unlocked: int | None,
        achievements_total: int | None,
        store_url: str,
    ) -> str:
        parts = [title]
        if hours_played is not None:
            parts.append(f"Hours played: {hours_played:g}")
        if last_played:
            parts.append(f"Last played: {last_played.isoformat()}")
        if achievements_unlocked is not None or achievements_total is not None:
            text = str(achievements_unlocked or 0)
            if achievements_total is not None:
                text = f"{text}/{achievements_total}"
            parts.append(f"Achievements: {text}")
        if store_url:
            parts.append(f"Store URL: {store_url}")
        return "\n".join(parts)

    def _source_id(self, app_id: str, fallback: str) -> str:
        if app_id:
            return f"steam_playtime_csv:{app_id}"
        digest = hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:24]
        return f"steam_playtime_csv:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _parse_hours(self, value: str) -> float | None:
        number = self._parse_float(value)
        return number if number is None else round(number, 2)

    def _parse_minutes(self, value: str) -> int | None:
        number = self._parse_float(value)
        return None if number is None else int(round(number))

    def _parse_int(self, value: str) -> int | None:
        number = self._parse_float(value)
        return None if number is None else int(round(number))

    def _parse_float(self, value: str) -> float | None:
        if not value:
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if not match:
            return None
        number = float(match.group(0))
        return number if number >= 0 else None

    def _split_values(self, value: str) -> list[str]:
        values: list[str] = []
        for item in re.split(r"[,;|]", value or ""):
            normalized = " ".join(item.casefold().split())
            if normalized and normalized not in values:
                values.append(normalized)
        return values

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
