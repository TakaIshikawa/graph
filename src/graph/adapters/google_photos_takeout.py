"""Adapter for Google Photos Takeout JSON sidecar metadata."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


VIDEO_EXTENSIONS = {".3gp", ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}


class GooglePhotosTakeoutAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_photos_takeout"

    @property
    def entity_types(self) -> list[str]:
        return ["album", "photo", "video"]

    def __init__(self, path: str = "", *, album: str = "", album_context: str = "") -> None:
        self.path = path
        self.album = album or album_context

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types or self.entity_types)
        if not requested.intersection(self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units_by_source_id: dict[str, KnowledgeUnit] = {}
        album_links: list[tuple[KnowledgeUnit, KnowledgeUnit]] = []
        root = Path(self.path).expanduser() if self.path else None
        album_units = self._album_units(root)
        album_by_dir = {album.metadata["album_dir"]: album for album in album_units}
        for path in self._iter_paths(root):
            if path.name == "metadata.json":
                continue
            try:
                parsed = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(parsed, dict):
                continue
            unit = self._unit_from_sidecar(parsed, path, root)
            if unit is None:
                continue
            if sync_at and unit.updated_at <= sync_at:
                continue
            units_by_source_id.setdefault(unit.source_id, unit)
            album_dir = path.parent.relative_to(root).as_posix() if root and root.is_dir() else ""
            album = album_by_dir.get(album_dir)
            if album is not None:
                album_links.append((album, units_by_source_id[unit.source_id]))

        if "album" in requested:
            for album in album_units:
                if not sync_at or album.updated_at > sync_at:
                    result.units.append(album)
        for unit in units_by_source_id.values():
            if unit.source_entity_type in requested:
                result.units.append(unit)
        if "album" in requested:
            for album, media in album_links:
                if media.source_entity_type in requested:
                    result.edges.append(self._album_edge(album, media))

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self, root: Path | None) -> list[Path]:
        if root is None or not root.exists():
            return []
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.json") if path.is_file())

    def _unit_from_sidecar(
        self,
        sidecar: dict[str, Any],
        path: Path,
        root: Path | None,
    ) -> KnowledgeUnit | None:
        title = self._first(sidecar, "title", "filename", "name") or path.stem.removesuffix(".json")
        description = self._first(sidecar, "description")
        taken_time = self._time_value(sidecar.get("photoTakenTime"))
        creation_time = self._time_value(sidecar.get("creationTime"))
        created_at = self._parse_datetime(taken_time) or self._parse_datetime(creation_time)
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        entity_type = "video" if Path(title).suffix.lower() in VIDEO_EXTENSIONS else "photo"
        source_file = self._relative_path(path, root)
        album = self.album or self._album_from_path(path, root)
        geo = self._geo_metadata(sidecar)
        people = self._people(sidecar.get("people"))

        metadata: dict[str, Any] = {
            "title": title,
            "description": description,
            "imageViews": self._parse_int(sidecar.get("imageViews")),
            "creationTime": self._normalized_time(sidecar.get("creationTime")),
            "photoTakenTime": self._normalized_time(sidecar.get("photoTakenTime")),
            "url": self._first(sidecar, "url"),
            "people": people,
            "source_file": source_file,
        }
        if album:
            metadata["album"] = album
        for key in ("geoData", "geoDataExif"):
            value = sidecar.get(key)
            if isinstance(value, dict):
                metadata[key] = value
        metadata.update(geo)

        tags = ["google_photos", entity_type]
        if album:
            tags.append(f"album:{album}")

        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_PHOTOS_TAKEOUT,
            source_id=self._source_id(sidecar, source_file, title, taken_time, creation_time),
            source_entity_type=entity_type,
            title=title,
            content=description or title,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=created_at,
            updated_at=self._parse_datetime(creation_time) or created_at,
        )

    def _source_id(
        self,
        sidecar: dict[str, Any],
        source_file: str,
        title: str,
        taken_time: str,
        creation_time: str,
    ) -> str:
        stable = self._first(sidecar, "url") or json.dumps(
            {
                "title": title,
                "photoTakenTime": taken_time,
                "creationTime": creation_time,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
        return f"google_photos_takeout:{digest}"

    def _album_units(self, root: Path | None) -> list[KnowledgeUnit]:
        albums: list[KnowledgeUnit] = []
        for path in self._album_metadata_paths(root):
            try:
                parsed = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(parsed, dict):
                continue
            album = self._album_unit(parsed, path, root)
            if album is not None:
                albums.append(album)
        return sorted(albums, key=lambda unit: unit.source_id)

    def _album_metadata_paths(self, root: Path | None) -> list[Path]:
        if root is None or not root.exists() or not root.is_dir():
            return []
        return sorted(path for path in root.rglob("metadata.json") if path.is_file())

    def _album_unit(self, metadata: dict[str, Any], path: Path, root: Path | None) -> KnowledgeUnit | None:
        album_dir = path.parent.relative_to(root).as_posix() if root and root.is_dir() else path.parent.name
        if album_dir in ("", "."):
            return None
        title = self._first(metadata, "title", "name") or Path(album_dir).name
        description = self._first(metadata, "description")
        date_text = self._time_value(metadata.get("date")) or self._time_value(metadata.get("creationTime"))
        created_at = self._parse_datetime(date_text) or datetime.now(timezone.utc)
        item_count = len([item for item in path.parent.glob("*.json") if item.name != "metadata.json"])
        source_file = self._relative_path(path, root)
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_PHOTOS_TAKEOUT,
            source_id=self._album_source_id(album_dir, title, date_text),
            source_entity_type="album",
            title=title,
            content=description or title,
            content_type=ContentType.METADATA,
            metadata={
                "title": title,
                "description": description,
                "date": date_text,
                "album_dir": album_dir,
                "path": album_dir,
                "item_count": item_count,
                "source_file": source_file,
            },
            tags=["google_photos", "album"],
            created_at=created_at,
            updated_at=created_at,
        )

    def _album_source_id(self, album_dir: str, title: str, date_text: str) -> str:
        digest = hashlib.sha256(
            json.dumps({"album_dir": album_dir, "title": title, "date": date_text}, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        return f"google_photos_takeout:album:{digest}"

    def _album_edge(self, album: KnowledgeUnit, media: KnowledgeUnit) -> KnowledgeEdge:
        digest = hashlib.sha256(f"{album.source_id}|{media.source_id}|contains".encode("utf-8")).hexdigest()[:24]
        return KnowledgeEdge(
            id=f"google-photos-album-contains-{digest}",
            from_unit_id=album.source_id,
            to_unit_id=media.source_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.GOOGLE_PHOTOS_TAKEOUT.value,
                "album": album.title,
                "album_dir": album.metadata.get("album_dir"),
                "media_title": media.title,
            },
            created_at=media.created_at,
        )

    def _geo_metadata(self, sidecar: dict[str, Any]) -> dict[str, Any]:
        for source_key in ("geoData", "geoDataExif"):
            value = sidecar.get(source_key)
            if not isinstance(value, dict):
                continue
            latitude = self._parse_float(value.get("latitude"))
            longitude = self._parse_float(value.get("longitude"))
            if latitude is None or longitude is None:
                continue
            if latitude == 0 and longitude == 0:
                continue
            geo = {
                "latitude": latitude,
                "longitude": longitude,
                "geo_source": source_key,
            }
            altitude = self._parse_float(value.get("altitude"))
            if altitude is not None:
                geo["altitude"] = altitude
            return geo
        return {}

    def _people(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        people: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                person = {key: item.get(key) for key in ("name", "url") if item.get(key)}
                if person:
                    people.append(person)
            elif str(item).strip():
                people.append({"name": str(item).strip()})
        return people

    def _time_value(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._first(value, "timestamp", "formatted")
        if value is None:
            return ""
        return str(value).strip()

    def _normalized_time(self, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, str] = {}
        for key in ("timestamp", "formatted"):
            if value.get(key) is not None and str(value[key]).strip():
                normalized[key] = str(value[key]).strip()
        return normalized

    def _parse_datetime(self, value: str) -> datetime | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        if raw.isdigit():
            return datetime.fromtimestamp(int(raw), tz=timezone.utc)
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return self._ensure_utc(parsed)

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _relative_path(self, path: Path, root: Path | None) -> str:
        if root and root.is_dir():
            return path.relative_to(root).as_posix()
        return path.name

    def _album_from_path(self, path: Path, root: Path | None) -> str:
        if root and root.is_dir():
            parent = path.parent.relative_to(root).as_posix()
            return "" if parent == "." else parent
        return path.parent.name

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
