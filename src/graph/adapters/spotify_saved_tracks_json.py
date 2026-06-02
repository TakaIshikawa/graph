"""Adapter for Spotify saved tracks JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime, parse_int
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SpotifySavedTracksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "spotify_saved_tracks_json"

    @property
    def entity_types(self) -> list[str]:
        return ["saved_track"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "saved_track" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._records(json.loads(path.read_text(encoding="utf-8-sig")))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records, start=1):
                unit = self._unit(record, path.name, index)
                if unit is None or (sync_at and unit.updated_at <= sync_at):
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("saved_tracks", "savedTracks", "items", "tracks", "data"):
                records = self._records(value.get(key))
                if records:
                    return records
            return [value]
        return []

    def _unit(self, record: dict[str, Any], source_file: str, source_row: int) -> KnowledgeUnit | None:
        track = self._dict(record.get("track"))
        source = track or record
        track_id = self._text(self._get(source, "id", "track_id", "trackId", "spotify_id", "spotifyId"))
        name = self._text(self._get(source, "name", "track_name", "trackName", "title"))
        artists = self._artists(self._get(source, "artists", "artist", "artist_name", "artistName"))
        album = self._album(source)
        added = parse_datetime(self._get(record, "added_at", "addedAt", "date_added", "dateAdded", "saved_at", "savedAt"))
        updated = parse_datetime(self._get(record, "updated_at", "updatedAt", "modified_at", "modifiedAt")) or added
        release_date = self._text(self._get(album, "release_date", "releaseDate"))
        duration_ms = parse_int(self._get(source, "duration_ms", "durationMs", "duration"))
        popularity = parse_int(self._get(source, "popularity"))
        explicit = self._bool(self._get(source, "explicit"))
        preview_url = self._text(self._get(source, "preview_url", "previewUrl"))
        url = self._url(source)
        isrc = self._text(self._get(self._dict(source.get("external_ids")), "isrc", "ISRC")) or self._text(self._get(source, "isrc", "ISRC"))
        if not any([track_id, name, artists, isrc]):
            return None

        metadata = clean_metadata(
            {
                "track_id": track_id,
                "name": name,
                "artists": artists,
                "album": self._text(self._get(album, "name", "album_name", "albumName", "title")),
                "album_id": self._text(self._get(album, "id", "album_id", "albumId")),
                "album_release_date": release_date,
                "added_at": added.isoformat() if added else self._text(self._get(record, "added_at", "addedAt", "date_added", "dateAdded")),
                "updated_at": updated.isoformat() if updated else self._text(self._get(record, "updated_at", "updatedAt", "modified_at", "modifiedAt")),
                "duration_ms": duration_ms,
                "popularity": popularity,
                "explicit": explicit,
                "preview_url": preview_url,
                "url": url,
                "source_url": url,
                "external_url": url,
                "isrc": isrc,
                "uri": self._text(self._get(source, "uri")),
                "source_file": source_file,
                "source_row": source_row,
            }
        )
        now = datetime.now(timezone.utc)
        title = name or track_id or "Spotify saved track"
        return KnowledgeUnit(
            source_project=self.name,
            source_id=f"{self.name}:{track_id}" if track_id else digest_source_id(self.name, isrc, name, ",".join(artists)),
            source_entity_type="saved_track",
            title=title,
            content=self._content(title, metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["spotify", "saved_track", *artists] if tag)),
            created_at=added or updated or now,
            updated_at=updated or added or now,
        )

    def _content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (("artists", "Artists"), ("album", "Album"), ("album_release_date", "Released"), ("added_at", "Added"), ("url", "URL")):
            if key in metadata:
                value = ", ".join(metadata[key]) if isinstance(metadata[key], list) else metadata[key]
                parts.append(f"{label}: {value}")
        return "\n".join(parts)

    def _album(self, source: dict[str, Any]) -> dict[str, Any]:
        album = source.get("album")
        if isinstance(album, dict):
            return album
        return {"name": album or self._get(source, "album_name", "albumName"), "release_date": self._get(source, "release_date", "releaseDate")}

    def _artists(self, value: Any) -> list[str]:
        if isinstance(value, list):
            artists = [self._text(item.get("name") if isinstance(item, dict) else item) for item in value]
            return list(dict.fromkeys(artist for artist in artists if artist))
        text = self._text(value)
        if not text:
            return []
        return list(dict.fromkeys(part.strip() for part in text.replace(";", ",").split(",") if part.strip()))

    def _url(self, source: dict[str, Any]) -> str:
        external_urls = self._dict(source.get("external_urls") or source.get("externalUrls"))
        return self._text(external_urls.get("spotify") or self._get(source, "external_url", "externalUrl", "url", "href"))

    def _get(self, record: dict[str, Any], *keys: str) -> Any:
        compact = {"".join(ch for ch in str(k).casefold() if ch.isalnum()): v for k, v in record.items()}
        for key in keys:
            if key in record:
                return record[key]
            value = compact.get("".join(ch for ch in key.casefold() if ch.isalnum()))
            if value is not None:
                return value
        return None

    def _dict(self, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()

    def _bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        text = self._text(value).casefold()
        if text in {"true", "yes", "1", "explicit"}:
            return True
        if text in {"false", "no", "0", "clean"}:
            return False
        return None
