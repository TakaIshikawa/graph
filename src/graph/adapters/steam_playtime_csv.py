"""Adapter for Steam playtime CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SteamPlaytimeCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "steam_playtime_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["game"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "game" not in entity_types:
            return result
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit:
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        app_id = first(row, "app id", "appid", "app_id")
        name = first(row, "game name", "name", "title")
        if not app_id and not name:
            return None
        minutes = _playtime_minutes(row)
        last_played = first(row, "last played", "last_played")
        platform = first(row, "platform")
        url = first(row, "store url", "url")
        last_at = parse_datetime(last_played) or datetime.now(timezone.utc)
        metadata = clean_metadata({"app_id": app_id, "playtime_minutes": minutes, "last_played": last_played, "platform": platform, "url": url, "external_url": url, "source_file": source_file})
        return KnowledgeUnit(source_project="steam_playtime_csv", source_id=f"steam_playtime_csv:{app_id}" if app_id else digest_source_id("steam_playtime_csv", name, index), source_entity_type="game", title=name or app_id, content=_content(name or app_id, minutes, last_played, platform, url), content_type=ContentType.ARTIFACT, metadata=metadata, created_at=last_at, updated_at=last_at)


def _playtime_minutes(row: dict[str, Any]) -> int | None:
    minutes = parse_float(first(row, "playtime minutes", "minutes", "playtime_min"))
    if minutes is not None:
        return int(round(minutes))
    hours = parse_float(first(row, "playtime hours", "hours", "playtime"))
    return int(round(hours * 60)) if hours is not None else None


def _content(title: str, minutes: int | None, last_played: str, platform: str, url: str) -> str:
    parts = [title]
    if minutes is not None:
        parts.append(f"Playtime minutes: {minutes}")
    for label, value in (("Last played", last_played), ("Platform", platform), ("URL", url)):
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(parts)
