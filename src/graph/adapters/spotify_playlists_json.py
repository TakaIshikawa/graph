"""Adapter for Spotify playlist JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class SpotifyPlaylistsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "spotify_playlists_json"

    @property
    def entity_types(self) -> list[str]:
        return ["playlist", "playlist_track"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or self.entity_types)
        if not allowed.intersection(self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        playlists: list[KnowledgeUnit] = []
        tracks: list[KnowledgeUnit] = []
        edges: list[KnowledgeEdge] = []
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._playlists(json.loads(path.read_text(encoding="utf-8-sig")))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for playlist_index, playlist in enumerate(records):
                playlist_unit = self._playlist_unit(playlist, path.name, playlist_index)
                if not playlist_unit:
                    continue
                playlist_tracks = []
                for track_index, item in enumerate(self._tracks(playlist)):
                    track_unit = self._track_unit(item, playlist_unit, path.name, track_index)
                    if track_unit and (sync_at is None or track_unit.updated_at > sync_at):
                        playlist_tracks.append(track_unit)
                        edges.append(KnowledgeEdge(id=digest_source_id(f"{self.name}:edge", playlist_unit.source_id, track_unit.source_id), from_unit_id=playlist_unit.source_id, to_unit_id=track_unit.source_id, relation=EdgeRelation.CONTAINS, source=EdgeSource.SOURCE, metadata={"source_project": self.name}))
                playlists.append(playlist_unit)
                tracks.extend(playlist_tracks)
        if "playlist" in allowed:
            result.units.extend(playlists)
        if "playlist_track" in allowed:
            result.units.extend(tracks)
        if {"playlist", "playlist_track"}.issubset(allowed):
            result.edges.extend({edge.id: edge for edge in edges}.values())
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _playlists(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("playlists", "items", "data"):
                if isinstance(value.get(key), (dict, list)):
                    records = self._playlists(value[key])
                    if records:
                        return records
            return [value] if first(value, "name", "title", "id") or self._tracks(value) else []
        return []

    def _playlist_unit(self, playlist: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        name = first(playlist, "name", "title")
        playlist_id = first(playlist, "id", "playlist_id", "uri")
        if not (name or playlist_id or self._tracks(playlist)):
            return None
        created_at = parse_datetime(first(playlist, "created_at", "createdAt"))
        updated_at = parse_datetime(first(playlist, "updated_at", "updatedAt", "modified_at")) or created_at or datetime.now(timezone.utc)
        description = first(playlist, "description")
        owner = self._owner(playlist.get("owner")) or first(playlist, "owner", "owner_name")
        collaborative = self._bool(playlist.get("collaborative"))
        metadata = clean_metadata({"playlist_id": playlist_id, "name": name, "description": description, "owner": owner, "collaborative": collaborative, "source_file": source_file})
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:playlist:{playlist_id}" if playlist_id else digest_source_id(f"{self.name}:playlist", name, index), source_entity_type="playlist", title=name or playlist_id or "Spotify playlist", content=self._playlist_content(name or playlist_id, description, owner), content_type=ContentType.METADATA, metadata=metadata, tags=["spotify", "playlist"], created_at=created_at or updated_at, updated_at=updated_at)

    def _track_unit(self, item: dict[str, Any], playlist: KnowledgeUnit, source_file: str, index: int) -> KnowledgeUnit | None:
        track = item.get("track") if isinstance(item.get("track"), dict) else item
        name = first(track, "name", "title", "trackName")
        artists = self._artists(track.get("artists") or first(track, "artist", "artistName"))
        album = self._album(track)
        uri = first(track, "uri", "track_uri")
        url = self._url(track) or first(item, "url")
        track_id = first(track, "id", "track_id") or uri
        if not any((name, artists, track_id, url)):
            return None
        added_at = parse_datetime(first(item, "added_at", "addedAt", "date_added")) or parse_datetime(first(track, "added_at", "addedAt"))
        updated_at = added_at or datetime.now(timezone.utc)
        metadata = clean_metadata({"playlist_source_id": playlist.source_id, "playlist_title": playlist.title, "track_id": track_id, "name": name, "artists": artists, "album": album, "uri": uri, "url": url, "added_at": added_at.isoformat() if added_at else None, "source_file": source_file})
        return KnowledgeUnit(source_project=self.name, source_id=digest_source_id(f"{self.name}:track", playlist.source_id, track_id or url or name, index), source_entity_type="playlist_track", title=name or track_id or "Spotify playlist track", content=self._track_content(name, artists, album, url), content_type=ContentType.ARTIFACT, metadata=metadata, tags=["spotify", "playlist_track", *artists], created_at=added_at or updated_at, updated_at=updated_at)

    def _tracks(self, playlist: dict[str, Any]) -> list[dict[str, Any]]:
        value = playlist.get("tracks")
        if isinstance(value, dict):
            value = value.get("items")
        if not isinstance(value, list):
            value = playlist.get("items") or playlist.get("entries")
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    def _artists(self, value: Any) -> list[str]:
        if isinstance(value, list):
            artists = []
            for item in value:
                artist = first(item, "name") if isinstance(item, dict) else str(item).strip()
                if artist:
                    artists.append(artist)
            return list(dict.fromkeys(artists))
        return [part.strip() for part in str(value or "").replace(";", ",").split(",") if part.strip()]

    def _album(self, track: dict[str, Any]) -> str:
        album = track.get("album")
        return first(album, "name", "title") if isinstance(album, dict) else str(album or first(track, "album", "albumName")).strip()

    def _owner(self, owner: Any) -> str:
        return first(owner, "display_name", "name", "id") if isinstance(owner, dict) else ""

    def _url(self, track: dict[str, Any]) -> str:
        urls = track.get("external_urls") if isinstance(track.get("external_urls"), dict) else {}
        return str(urls.get("spotify") or first(track, "url", "href")).strip()

    def _bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().casefold()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        return None

    def _playlist_content(self, name: str, description: str, owner: str) -> str:
        return "\n".join(part for part in (name, description, f"Owner: {owner}" if owner else "") if part)

    def _track_content(self, name: str, artists: list[str], album: str, url: str) -> str:
        return "\n".join(part for part in (name, f"Artists: {', '.join(artists)}" if artists else "", f"Album: {album}" if album else "", f"URL: {url}" if url else "") if part)
