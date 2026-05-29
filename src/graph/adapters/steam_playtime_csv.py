"""Adapter for manually exported Steam playtime CSV files."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, first, iter_paths, parse_datetime, parse_float, read_csv_rows
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

        sync_at = _ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
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

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        app_id = first(row, "app_id", "appid", "app id", "AppID", "Application ID", "steam_appid", "Game ID")
        game_name = first(row, "game", "game name", "name", "title", "Game", "Name", "Title")
        store_url = first(row, "store_url", "store url", "url", "URL", "link", "Store URL")
        explicit_url = store_url
        if not app_id and not game_name and not store_url:
            return None

        hours_played = _parse_hours(first(row, "hours_played", "Hours Played", "hours", "Hours", "playtime hours", "playtime"))
        minutes_played = _parse_minutes(first(row, "minutes_played", "Minutes Played", "playtime_forever", "playtime minutes", "minutes", "playtime_min"))
        if hours_played is None and minutes_played is not None:
            hours_played = round(minutes_played / 60, 2)
        playtime_minutes = minutes_played
        if playtime_minutes is None and hours_played is not None:
            playtime_minutes = int(round(hours_played * 60))

        last_played_text = first(row, "last_played", "Last Played", "last played", "last_played_at")
        first_played_text = first(row, "first_played", "First Played", "first played", "first_played_at")
        last_played = parse_datetime(last_played_text)
        first_played = parse_datetime(first_played_text)
        achievements_unlocked = _parse_int(first(row, "achievements_unlocked", "Achievements Unlocked", "unlocked"))
        achievements_total = _parse_int(first(row, "achievements_total", "Achievements Total", "total achievements", "total"))
        platform = first(row, "platform", "Platform")
        platforms = _split_values(first(row, "platforms", "Platforms", "owned_platforms", "Owned Platforms") or platform)
        if not store_url and app_id:
            store_url = f"https://store.steampowered.com/app/{app_id}/"

        now = datetime.now(timezone.utc)
        title = game_name or f"Steam app {app_id}"
        metadata = clean_metadata(
            {
                "app_id": app_id,
                "game_name": game_name,
                "hours_played": hours_played,
                "minutes_played": minutes_played,
                "playtime_minutes": playtime_minutes,
                "last_played": last_played.isoformat() if last_played else last_played_text,
                "first_played": first_played.isoformat() if first_played else first_played_text,
                "achievements_unlocked": achievements_unlocked,
                "achievements_total": achievements_total,
                "store_url": store_url,
                "url": explicit_url,
                "external_url": explicit_url,
                "platform": platform,
                "platforms": platforms,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project=self.name,
            source_id=self._source_id(app_id, game_name or store_url),
            source_entity_type="game_playtime",
            title=title,
            content=_content(title, hours_played, last_played, achievements_unlocked, achievements_total, store_url, platform),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["steam", "game_playtime", *platforms],
            created_at=first_played or last_played or now,
            updated_at=last_played or first_played or now,
        )

    def _source_id(self, app_id: str, fallback: str) -> str:
        if app_id:
            return f"steam_playtime_csv:{app_id}"
        digest = hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:24]
        return f"steam_playtime_csv:{digest}"


def _content(
    title: str,
    hours_played: float | None,
    last_played: datetime | None,
    achievements_unlocked: int | None,
    achievements_total: int | None,
    store_url: str,
    platform: str,
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
    if platform:
        parts.append(f"Platform: {platform}")
    if store_url:
        parts.append(f"Store URL: {store_url}")
    return "\n".join(parts)


def _parse_hours(value: str) -> float | None:
    number = _parse_nonnegative_float(value)
    return number if number is None else round(number, 2)


def _parse_minutes(value: str) -> int | None:
    number = _parse_nonnegative_float(value)
    return None if number is None else int(round(number))


def _parse_int(value: str) -> int | None:
    number = _parse_nonnegative_float(value)
    return None if number is None else int(round(number))


def _parse_nonnegative_float(value: str) -> float | None:
    number = parse_float(value)
    return number if number is not None and number >= 0 else None


def _split_values(value: str) -> list[str]:
    values: list[str] = []
    for item in value.replace("|", ";").replace(",", ";").split(";"):
        normalized = " ".join(item.casefold().split())
        if normalized and normalized not in values:
            values.append(normalized)
    return values


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
