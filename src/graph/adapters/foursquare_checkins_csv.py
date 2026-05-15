"""Adapter for Foursquare and Swarm checkins CSV exports."""

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


class FoursquareCheckinsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "foursquare_checkins_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["checkin"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "checkin" not in set(entity_types or self.entity_types):
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
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
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
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        venue = self._first(row, "Venue", "Name", "Venue Name", "Place")
        category = self._first(row, "Category", "Venue Category")
        city = self._first(row, "City", "Town")
        address = self._first(row, "Address", "Street Address", "Location")
        created_at = self._parse_datetime(self._first(row, "Created At", "Created", "Date", "Checkin Date", "Time"))
        note = self._first(row, "Shout", "Comment", "Note", "Notes")
        url = self._first(row, "URL", "Checkin URL", "Venue URL", "Link")
        latitude = self._parse_coordinate(self._first(row, "Latitude", "Lat"), -90, 90)
        longitude = self._parse_coordinate(self._first(row, "Longitude", "Lng", "Lon", "Long"), -180, 180)
        if not venue and not url:
            return None
        metadata = {
            "venue": venue,
            "category": category,
            "city": city,
            "address": address,
            "created_at": created_at.isoformat() if created_at else self._first(row, "Created At", "Created", "Date", "Checkin Date", "Time"),
            "note": note,
            "url": url,
            "latitude": latitude,
            "longitude": longitude,
            "source_file": source_file,
            "row": dict(row),
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.FOURSQUARE_CHECKINS_CSV,
            source_id=self._source_id(venue, created_at, url, city),
            source_entity_type="checkin",
            title=venue or url,
            content=self._content(metadata),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(item for item in ["foursquare", "checkin", category, city] if item)),
            created_at=created_at or now,
            updated_at=created_at or now,
        )

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [metadata.get("venue")]
        for key, label in (("category", "Category"), ("city", "City"), ("address", "Address"), ("created_at", "Checked in"), ("note", "Note"), ("url", "URL")):
            if metadata.get(key):
                parts.append(f"{label}: {metadata[key]}")
        if metadata.get("latitude") is not None and metadata.get("longitude") is not None:
            parts.append(f"Coordinates: {metadata['latitude']}, {metadata['longitude']}")
        return "\n".join(str(item) for item in parts if item)

    def _source_id(self, venue: str, created_at: datetime | None, url: str, city: str) -> str:
        raw = url or "|".join([self._normalized(venue), self._normalized(city), created_at.isoformat() if created_at else ""])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"foursquare_checkins_csv:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_coordinate(self, value: str, minimum: float, maximum: float) -> float | None:
        if not value:
            return None
        try:
            number = float(value.strip())
        except ValueError:
            return None
        if minimum <= number <= maximum:
            return number
        return None

    def _parse_datetime(self, value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        for candidate in (text, text.replace("Z", "+00:00")):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate))
            except ValueError:
                pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _normalized(self, value: str) -> str:
        return " ".join(value.casefold().split())

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
