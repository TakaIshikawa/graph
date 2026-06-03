"""Adapter for Google Photos favorite media metadata JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GooglePhotosFavoritesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_photos_favorites_json"

    @property
    def entity_types(self) -> list[str]:
        return ["media"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "media" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                parsed = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            favorites_export = "favorite" in path.stem.casefold() or "favourite" in path.stem.casefold()
            for index, record in enumerate(_records(parsed)):
                if not favorites_export and not _is_favorite(record):
                    continue
                unit = self._unit(record, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = _text(record.get("title") or record.get("filename") or record.get("name"))
        origin = record.get("googlePhotosOrigin") if isinstance(record.get("googlePhotosOrigin"), dict) else {}
        url = _text(record.get("url") or origin.get("url"))
        description = _text(record.get("description"))
        if not any((title, url, description)):
            return None
        taken_at = _time(record.get("photoTakenTime")) or parse_datetime(record.get("photo_taken_time"))
        created_at = _time(record.get("creationTime")) or parse_datetime(record.get("creation_time") or record.get("created_at"))
        mime_type = _text(record.get("mimeType") or record.get("mime_type"))
        media_type = _text(record.get("mediaType") or record.get("media_type")) or ("video" if mime_type.startswith("video/") else "photo")
        people = [_text(person.get("name") if isinstance(person, dict) else person) for person in record.get("people", [])]
        people = [person for person in dict.fromkeys(people) if person]
        geo = _geo(record.get("geoData") or record.get("geo_data"))
        album = _text(record.get("album") or record.get("albumTitle") or record.get("album_title"))
        metadata = clean_metadata(
            {
                "title": title,
                "filename": _text(record.get("filename") or title),
                "description": description,
                "url": url,
                "media_type": media_type,
                "photo_taken_time": taken_at.isoformat() if taken_at else None,
                "creation_time": created_at.isoformat() if created_at else None,
                "geo_data": geo or None,
                "people": people,
                "album": album,
                "favorite": True,
                "source_file": source_file,
            }
        )
        now = datetime.now(timezone.utc)
        timestamp = taken_at or created_at
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_PHOTOS_FAVORITES_JSON,
            source_id=digest_source_id("google_photos_favorites_json", title, url, timestamp.isoformat() if timestamp else "", source_file, index),
            source_entity_type="media",
            title=title or "Google Photos favorite",
            content=_content(title, description, metadata),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=[tag for tag in dict.fromkeys(["google_photos", "favorite", media_type, album]) if tag],
            created_at=timestamp or now,
            updated_at=created_at or taken_at or now,
        )


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("media", "items", "photos", "data", "results"):
            items = value.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return [value]
    return []


def _is_favorite(record: dict[str, Any]) -> bool:
    for key in ("favorite", "favorited", "isFavorite", "is_favorite"):
        value = record.get(key)
        if isinstance(value, bool):
            return value
        if _text(value).casefold() in {"1", "true", "yes", "favorite", "favorited"}:
            return True
    return False


def _time(value: Any) -> datetime | None:
    if isinstance(value, dict):
        return parse_datetime(value.get("timestamp") or value.get("formatted"))
    return parse_datetime(value)


def _geo(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return clean_metadata({"latitude": value.get("latitude"), "longitude": value.get("longitude"), "altitude": value.get("altitude")})


def _content(title: str, description: str, metadata: dict[str, Any]) -> str:
    parts = [title, description]
    for key, label in (("album", "Album"), ("photo_taken_time", "Taken"), ("url", "URL")):
        if metadata.get(key):
            parts.append(f"{label}: {metadata[key]}")
    return "\n".join(part for part in parts if part)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()
