"""Adapter for Spotify Takeout streaming history exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class SpotifyStreamingHistoryAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "spotify_streaming_history"

    @property
    def entity_types(self) -> list[str]:
        return ["artist", "track", "play", "session", "podcast_show", "podcast_episode"]

    def __init__(self, path: str = "", session_gap_minutes: int = 30) -> None:
        self.path = path
        self.session_gap = timedelta(minutes=session_gap_minutes)

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = self._sync_datetime(since) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                items = self._read_items(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue

            for item in items:
                unit = self._unit_from_item(item, path.name)
                if unit is None:
                    continue
                if sync_at and unit.created_at <= sync_at:
                    continue
                units.append(unit)

        units = sorted(units, key=lambda unit: (unit.created_at, unit.source_id))
        artists = self._artist_units(units) if "artist" in allowed_types else []
        tracks = self._track_units(units) if "track" in allowed_types else []
        sessions = self._session_units(units) if "session" in allowed_types else []
        podcast_shows = self._podcast_show_units(units) if "podcast_show" in allowed_types else []
        podcast_episodes = self._podcast_episode_units(units) if "podcast_episode" in allowed_types else []
        if "artist" in allowed_types:
            result.units.extend(artists)
        if "track" in allowed_types:
            result.units.extend(tracks)
        if "play" in allowed_types:
            result.units.extend(units)
        if "session" in allowed_types:
            result.units.extend(sessions)
        if "podcast_show" in allowed_types:
            result.units.extend(podcast_shows)
        if "podcast_episode" in allowed_types:
            result.units.extend(podcast_episodes)
        if "artist" in allowed_types and "track" in allowed_types:
            result.edges.extend(self._artist_track_edges(artists, tracks))
        if "track" in allowed_types and "play" in allowed_types:
            result.edges.extend(self._track_play_edges(tracks, units))
        if "play" in allowed_types and "session" in allowed_types:
            result.edges.extend(self._contains_edges(sessions))
        if "podcast_show" in allowed_types and "podcast_episode" in allowed_types:
            result.edges.extend(self._podcast_show_episode_edges(podcast_shows, podcast_episodes))
        if "podcast_episode" in allowed_types and "play" in allowed_types:
            result.edges.extend(self._podcast_episode_play_edges(podcast_episodes, units))
        if "podcast_show" in allowed_types and "play" in allowed_types:
            result.edges.extend(self._podcast_show_play_edges(podcast_shows, units))
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []

        root = Path(self.path).expanduser()
        if root.is_file() and self._is_history_file(root):
            return [root]
        if not root.is_dir():
            return []

        return sorted(
            (child for child in root.iterdir() if child.is_file() and self._is_history_file(child)),
            key=lambda child: child.name,
        )

    def _is_history_file(self, path: Path) -> bool:
        name = path.name
        return bool(
            re.fullmatch(r"StreamingHistory_music_.*\.json", name)
            or re.fullmatch(r"endsong_.*\.json", name)
        )

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("items", "history", "plays", "data"):
                nested = parsed.get(key)
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_item(self, item: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        timestamp_text = self._first(item, "ts", "endTime", "end_time", "played_at")
        played_at = self._parse_datetime(timestamp_text)
        if played_at is None:
            return None

        track_name = self._first(
            item,
            "master_metadata_track_name",
            "trackName",
            "track_name",
            "song",
        )
        artist_name = self._first(
            item,
            "master_metadata_album_artist_name",
            "artistName",
            "artist_name",
            "artist",
        )
        album_name = self._first(
            item,
            "master_metadata_album_album_name",
            "albumName",
            "album_name",
            "album",
        )
        spotify_uri = self._first(item, "spotify_track_uri", "spotify_uri", "uri")
        episode_name = self._first(item, "episode_name", "master_metadata_episode_name")
        show_name = self._first(item, "episode_show_name", "master_metadata_show_name", "show_name")
        episode_uri = self._first(item, "spotify_episode_uri", "episode_uri")
        show_uri = self._first(item, "spotify_show_uri", "show_uri")
        ms_played = self._parse_int(self._value(item, "ms_played", "msPlayed", "ms_played"))

        is_podcast = bool(episode_name or show_name or episode_uri or show_uri)
        if not track_name and not artist_name and not spotify_uri and not is_podcast:
            return None

        title = self._podcast_title(episode_name, show_name) if is_podcast else self._title(track_name, artist_name)
        metadata = {
            "track_name": track_name,
            "artist_name": artist_name,
            "album_name": album_name,
            "spotify_uri": spotify_uri,
            "episode_name": episode_name,
            "show_name": show_name,
            "spotify_episode_uri": episode_uri,
            "spotify_show_uri": show_uri,
            "media_kind": "podcast" if is_podcast else "music",
            "ms_played": ms_played,
            "platform": self._first(item, "platform"),
            "country": self._first(item, "conn_country", "country"),
            "reason_start": self._first(item, "reason_start"),
            "reason_end": self._first(item, "reason_end"),
            "shuffle": self._parse_bool(self._value(item, "shuffle")),
            "skipped": self._parse_bool(self._value(item, "skipped")),
            "offline": self._parse_bool(self._value(item, "offline")),
            "source_file": source_file,
        }

        return KnowledgeUnit(
            source_project=SourceProject.SPOTIFY_STREAMING_HISTORY,
            source_id=self._source_id(played_at, episode_name or track_name, show_name or artist_name, ms_played),
            source_entity_type="play",
            title=title,
            content=self._podcast_content(episode_name, show_name, ms_played) if is_podcast else self._content(track_name, artist_name, album_name, ms_played),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["spotify", "podcast", "listening"] if is_podcast else ["spotify", "music", "listening"],
            created_at=played_at,
            updated_at=played_at,
        )

    def _source_id(
        self, played_at: datetime, track_name: str, artist_name: str, ms_played: int | None
    ) -> str:
        identifier = "|".join(
            [
                played_at.isoformat(),
                track_name.strip().lower(),
                artist_name.strip().lower(),
                "" if ms_played is None else str(ms_played),
            ]
        )
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:24]
        return f"spotify_streaming_history:{digest}"

    def _title(self, track_name: str, artist_name: str) -> str:
        if track_name and artist_name:
            return f"{track_name} - {artist_name}"
        return track_name or artist_name or "Spotify play"

    def _content(
        self, track_name: str, artist_name: str, album_name: str, ms_played: int | None
    ) -> str:
        parts = []
        if track_name:
            parts.append(f"Track: {track_name}")
        if artist_name:
            parts.append(f"Artist: {artist_name}")
        if album_name:
            parts.append(f"Album: {album_name}")
        if ms_played is not None:
            parts.append(f"Milliseconds played: {ms_played}")
        return "\n".join(parts)

    def _podcast_title(self, episode_name: str, show_name: str) -> str:
        if episode_name and show_name:
            return f"{episode_name} - {show_name}"
        return episode_name or show_name or "Spotify podcast play"

    def _podcast_content(self, episode_name: str, show_name: str, ms_played: int | None) -> str:
        parts = []
        if episode_name:
            parts.append(f"Episode: {episode_name}")
        if show_name:
            parts.append(f"Show: {show_name}")
        if ms_played is not None:
            parts.append(f"Milliseconds played: {ms_played}")
        return "\n".join(parts)

    def _session_units(self, plays: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        sessions: list[list[KnowledgeUnit]] = []
        for play in plays:
            if not sessions:
                sessions.append([play])
                continue
            previous = sessions[-1][-1]
            if play.created_at - previous.created_at <= self.session_gap:
                sessions[-1].append(play)
            else:
                sessions.append([play])

        return [self._session_unit(session) for session in sessions if session]

    def _artist_units(self, plays: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for play in plays:
            artist = str(play.metadata.get("artist_name") or "").strip()
            if not artist:
                continue
            key = artist.casefold()
            names.setdefault(key, artist)
            grouped.setdefault(key, []).append(play)
        return [
            self._aggregate_unit(
                "artist",
                self._artist_source_id(names[key]),
                names[key],
                artist_plays,
                {
                    "artist_name": names[key],
                    "track_names": sorted({str(play.metadata.get("track_name")) for play in artist_plays if play.metadata.get("track_name")}),
                },
                ["spotify", "music", "artist"],
            )
            for key, artist_plays in grouped.items()
        ]

    def _track_units(self, plays: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[tuple[str, str, str], list[KnowledgeUnit]] = {}
        for play in plays:
            key = self._track_key(play)
            if key:
                grouped.setdefault(key, []).append(play)
        units: list[KnowledgeUnit] = []
        for key, track_plays in grouped.items():
            first = track_plays[0]
            track_name = str(first.metadata.get("track_name") or "")
            artist_name = str(first.metadata.get("artist_name") or "")
            spotify_uri = str(first.metadata.get("spotify_uri") or "")
            units.append(
                self._aggregate_unit(
                    "track",
                    self._track_source_id(track_name, artist_name, spotify_uri),
                    self._title(track_name, artist_name),
                    track_plays,
                    {
                        "track_name": track_name,
                        "artist_name": artist_name,
                        "spotify_uri": spotify_uri,
                    },
                    ["spotify", "music", "track"],
                )
            )
        return units

    def _podcast_show_units(self, plays: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        uris: dict[str, str] = {}
        for play in plays:
            key = self._podcast_show_key(play)
            if not key:
                continue
            show_name = str(play.metadata.get("show_name") or "")
            show_uri = str(play.metadata.get("spotify_show_uri") or "")
            names.setdefault(key, show_name)
            uris.setdefault(key, show_uri)
            grouped.setdefault(key, []).append(play)
        return [
            self._aggregate_unit(
                "podcast_show",
                self._podcast_show_source_id(names[key], uris[key]),
                names[key] or uris[key],
                show_plays,
                {
                    "show_name": names[key],
                    "spotify_show_uri": uris[key],
                    "episode_names": sorted({str(play.metadata.get("episode_name")) for play in show_plays if play.metadata.get("episode_name")}),
                },
                ["spotify", "podcast", "show"],
            )
            for key, show_plays in grouped.items()
        ]

    def _podcast_episode_units(self, plays: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        first_by_key: dict[str, KnowledgeUnit] = {}
        for play in plays:
            key = self._podcast_episode_key(play)
            if not key:
                continue
            first_by_key.setdefault(key, play)
            grouped.setdefault(key, []).append(play)
        units: list[KnowledgeUnit] = []
        for key, episode_plays in grouped.items():
            first = first_by_key[key]
            episode_name = str(first.metadata.get("episode_name") or "")
            show_name = str(first.metadata.get("show_name") or "")
            episode_uri = str(first.metadata.get("spotify_episode_uri") or "")
            show_uri = str(first.metadata.get("spotify_show_uri") or "")
            units.append(
                self._aggregate_unit(
                    "podcast_episode",
                    self._podcast_episode_source_id(episode_name, show_name, episode_uri),
                    self._podcast_title(episode_name, show_name),
                    episode_plays,
                    {
                        "episode_name": episode_name,
                        "show_name": show_name,
                        "spotify_episode_uri": episode_uri,
                        "spotify_show_uri": show_uri,
                    },
                    ["spotify", "podcast", "episode"],
                )
            )
        return units

    def _aggregate_unit(
        self,
        entity_type: str,
        source_id: str,
        title: str,
        plays: list[KnowledgeUnit],
        extra_metadata: dict[str, Any],
        tags: list[str],
    ) -> KnowledgeUnit:
        total_ms = sum(int(play.metadata["ms_played"]) for play in plays if isinstance(play.metadata.get("ms_played"), int))
        first_played = min(play.created_at for play in plays)
        last_played = max(play.created_at for play in plays)
        metadata = {
            **extra_metadata,
            "play_count": len(plays),
            "total_ms_played": total_ms,
            "first_played_at": first_played.isoformat(),
            "last_played_at": last_played.isoformat(),
            "albums": sorted({str(play.metadata.get("album_name")) for play in plays if play.metadata.get("album_name")}),
            "countries": sorted({str(play.metadata.get("country")) for play in plays if play.metadata.get("country")}),
            "source_files": sorted({str(play.metadata.get("source_file")) for play in plays if play.metadata.get("source_file")}),
            "play_source_ids": [play.source_id for play in plays],
        }
        return KnowledgeUnit(
            source_project=SourceProject.SPOTIFY_STREAMING_HISTORY,
            source_id=source_id,
            source_entity_type=entity_type,
            title=title,
            content=f"Spotify {entity_type}: {title}",
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=tags,
            created_at=first_played,
            updated_at=last_played,
        )

    def _session_unit(self, plays: list[KnowledgeUnit]) -> KnowledgeUnit:
        start = min(play.created_at for play in plays)
        end = max(play.created_at for play in plays)
        artists = {str(play.metadata.get("artist_name") or "") for play in plays if play.metadata.get("artist_name")}
        tracks = {str(play.metadata.get("track_name") or "") for play in plays if play.metadata.get("track_name")}
        total_ms = sum(int(play.metadata["ms_played"]) for play in plays if isinstance(play.metadata.get("ms_played"), int))
        source_files = sorted({str(play.metadata.get("source_file")) for play in plays if play.metadata.get("source_file")})
        source_id = self._session_source_id(plays)
        return KnowledgeUnit(
            source_project=SourceProject.SPOTIFY_STREAMING_HISTORY,
            source_id=source_id,
            source_entity_type="session",
            title=f"Spotify session {start.isoformat()}",
            content=f"{len(plays)} Spotify plays from {start.isoformat()} to {end.isoformat()}",
            content_type=ContentType.METADATA,
            metadata={
                "start_at": start.isoformat(),
                "end_at": end.isoformat(),
                "play_count": len(plays),
                "total_ms_played": total_ms,
                "artist_count": len(artists),
                "track_count": len(tracks),
                "source_files": source_files,
                "play_source_ids": [play.source_id for play in plays],
            },
            tags=["spotify", "music", "listening-session"],
            created_at=start,
            updated_at=end,
        )

    def _contains_edges(self, sessions: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for session in sessions:
            for play_source_id in session.metadata.get("play_source_ids") or []:
                edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(session.source_id, str(play_source_id)),
                        from_unit_id=session.source_id,
                        to_unit_id=str(play_source_id),
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.SPOTIFY_STREAMING_HISTORY.value,
                            "relation_type": "session_contains_play",
                        },
                    )
                )
        return edges

    def _artist_track_edges(self, artists: list[KnowledgeUnit], tracks: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        artist_ids = {str(artist.metadata.get("artist_name") or "").casefold(): artist.source_id for artist in artists}
        edges: list[KnowledgeEdge] = []
        for track in tracks:
            artist_id = artist_ids.get(str(track.metadata.get("artist_name") or "").casefold())
            if artist_id:
                edges.append(self._edge(artist_id, track.source_id, "artist_contains_track"))
        return list({edge.id: edge for edge in edges}.values())

    def _podcast_show_episode_edges(self, shows: list[KnowledgeUnit], episodes: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        show_ids = {self._podcast_show_source_id(str(show.metadata.get("show_name") or ""), str(show.metadata.get("spotify_show_uri") or "")): show.source_id for show in shows}
        edges: list[KnowledgeEdge] = []
        for episode in episodes:
            show_id = show_ids.get(self._podcast_show_source_id(str(episode.metadata.get("show_name") or ""), str(episode.metadata.get("spotify_show_uri") or "")))
            if show_id:
                edges.append(self._edge(show_id, episode.source_id, "podcast_show_contains_episode"))
        return list({edge.id: edge for edge in edges}.values())

    def _podcast_episode_play_edges(self, episodes: list[KnowledgeUnit], plays: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        episode_ids = {
            self._podcast_episode_source_id(
                str(episode.metadata.get("episode_name") or ""),
                str(episode.metadata.get("show_name") or ""),
                str(episode.metadata.get("spotify_episode_uri") or ""),
            ): episode.source_id
            for episode in episodes
        }
        edges: list[KnowledgeEdge] = []
        for play in plays:
            episode_id = episode_ids.get(self._podcast_episode_source_id(str(play.metadata.get("episode_name") or ""), str(play.metadata.get("show_name") or ""), str(play.metadata.get("spotify_episode_uri") or "")))
            if episode_id:
                edges.append(self._edge(episode_id, play.source_id, "podcast_episode_contains_play"))
        return list({edge.id: edge for edge in edges}.values())

    def _podcast_show_play_edges(self, shows: list[KnowledgeUnit], plays: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        show_ids = {self._podcast_show_source_id(str(show.metadata.get("show_name") or ""), str(show.metadata.get("spotify_show_uri") or "")): show.source_id for show in shows}
        edges: list[KnowledgeEdge] = []
        for play in plays:
            show_id = show_ids.get(self._podcast_show_source_id(str(play.metadata.get("show_name") or ""), str(play.metadata.get("spotify_show_uri") or "")))
            if show_id:
                edges.append(self._edge(show_id, play.source_id, "podcast_show_contains_play"))
        return list({edge.id: edge for edge in edges}.values())

    def _track_play_edges(self, tracks: list[KnowledgeUnit], plays: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        track_ids = {
            self._track_source_id(
                str(track.metadata.get("track_name") or ""),
                str(track.metadata.get("artist_name") or ""),
                str(track.metadata.get("spotify_uri") or ""),
            ): track.source_id
            for track in tracks
        }
        edges: list[KnowledgeEdge] = []
        for play in plays:
            track_id = track_ids.get(
                self._track_source_id(
                    str(play.metadata.get("track_name") or ""),
                    str(play.metadata.get("artist_name") or ""),
                    str(play.metadata.get("spotify_uri") or ""),
                )
            )
            if track_id:
                edges.append(self._edge(track_id, play.source_id, "track_contains_play"))
        return list({edge.id: edge for edge in edges}.values())

    def _edge(self, from_id: str, to_id: str, relation_type: str) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=self._edge_id(from_id, to_id, relation_type),
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.SPOTIFY_STREAMING_HISTORY.value,
                "relation_type": relation_type,
            },
        )

    def _session_source_id(self, plays: list[KnowledgeUnit]) -> str:
        payload = "|".join(play.source_id for play in plays)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"spotify_streaming_history:session:{digest}"

    def _artist_source_id(self, artist_name: str) -> str:
        digest = hashlib.sha256(artist_name.strip().casefold().encode("utf-8")).hexdigest()[:24]
        return f"spotify_streaming_history:artist:{digest}"

    def _track_source_id(self, track_name: str, artist_name: str, spotify_uri: str) -> str:
        raw = spotify_uri.strip().casefold() or "|".join((track_name.strip().casefold(), artist_name.strip().casefold()))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"spotify_streaming_history:track:{digest}"

    def _podcast_show_source_id(self, show_name: str, show_uri: str) -> str:
        raw = show_uri.strip().casefold() or show_name.strip().casefold()
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"spotify_streaming_history:podcast_show:{digest}"

    def _podcast_episode_source_id(self, episode_name: str, show_name: str, episode_uri: str) -> str:
        raw = episode_uri.strip().casefold() or "|".join((episode_name.strip().casefold(), show_name.strip().casefold()))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"spotify_streaming_history:podcast_episode:{digest}"

    def _track_key(self, play: KnowledgeUnit) -> tuple[str, str, str] | None:
        if play.metadata.get("media_kind") == "podcast":
            return None
        track_name = str(play.metadata.get("track_name") or "")
        artist_name = str(play.metadata.get("artist_name") or "")
        spotify_uri = str(play.metadata.get("spotify_uri") or "")
        if not track_name and not spotify_uri:
            return None
        return (spotify_uri.casefold(), track_name.casefold(), artist_name.casefold())

    def _podcast_show_key(self, play: KnowledgeUnit) -> str | None:
        show_name = str(play.metadata.get("show_name") or "")
        show_uri = str(play.metadata.get("spotify_show_uri") or "")
        raw = show_uri.strip().casefold() or show_name.strip().casefold()
        return raw or None

    def _podcast_episode_key(self, play: KnowledgeUnit) -> str | None:
        episode_name = str(play.metadata.get("episode_name") or "")
        show_name = str(play.metadata.get("show_name") or "")
        episode_uri = str(play.metadata.get("spotify_episode_uri") or "")
        raw = episode_uri.strip().casefold() or "|".join((episode_name.strip().casefold(), show_name.strip().casefold()))
        return raw if episode_name or episode_uri else None

    def _edge_id(self, from_source_id: str, to_source_id: str, relation_type: str = "contains") -> str:
        digest = hashlib.sha256("|".join((from_source_id, to_source_id, relation_type)).encode("utf-8")).hexdigest()[:24]
        return f"spotify-streaming-history-{relation_type}-{digest}"

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        value = self._value(item, *keys)
        if value is None or isinstance(value, dict | list):
            return ""
        return str(value).strip()

    def _value(self, item: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in item:
                return item[key]
        return None

    def _parse_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(float(str(value).strip()))
        except ValueError:
            return None

    def _parse_bool(self, value: Any) -> bool | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "t", "yes", "y", "1"}:
            return True
        if text in {"false", "f", "no", "n", "0"}:
            return False
        return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return self._ensure_utc(value)

        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None

        return self._ensure_utc(parsed)

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        return self._ensure_utc(since.last_sync_at)
