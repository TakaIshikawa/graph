"""Adapter for iNaturalist observation CSV exports."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class INaturalistObservationsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "inaturalist_observations_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["observation"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "observation" not in entity_types:
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _iter_paths(self) -> list[Path]:
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if root.is_dir():
            return sorted(child for child in root.rglob("*.csv") if child.is_file())
        return []

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader]

    def _unit_from_row(self, row: dict[str, Any], path: Path) -> KnowledgeUnit | None:
        observation_id = self._first(row, "id", "Observation ID", "observation_id")
        common_name = self._first(row, "common_name", "Common Name", "taxon_common_name")
        scientific_name = self._first(row, "scientific_name", "Scientific Name", "taxon_scientific_name")
        if not observation_id and not common_name and not scientific_name:
            return None

        iconic_taxon = self._first(row, "iconic_taxon_name", "Iconic Taxon", "iconic_taxon")
        quality_grade = self._first(row, "quality_grade", "Quality Grade")
        place_guess = self._first(row, "place_guess", "Place Guess", "Location")
        latitude_raw = self._first(row, "latitude", "Latitude", "lat")
        longitude_raw = self._first(row, "longitude", "Longitude", "lng", "lon")
        geoprivacy = self._first(row, "geoprivacy", "Geoprivacy")
        url = self._first(row, "url", "URL", "uri")
        description = self._first(row, "description", "Description", "notes")
        tags = self._split_tags(self._first(row, "tags", "Tags"))
        observed_at = self._parse_datetime(self._first(row, "observed_on", "time_observed_at", "Observed Date", "observed_at"))
        created_at = self._parse_datetime(self._first(row, "created_at", "Created At", "created"))
        updated_at = self._parse_datetime(self._first(row, "updated_at", "Updated At", "updated"))
        comparable_at = updated_at or created_at or observed_at or datetime.now(timezone.utc)

        metadata = {
            "observation_id": observation_id,
            "observed_at": observed_at.isoformat() if observed_at else "",
            "created_at": created_at.isoformat() if created_at else "",
            "updated_at": updated_at.isoformat() if updated_at else "",
            "common_name": common_name,
            "scientific_name": scientific_name,
            "iconic_taxon": iconic_taxon,
            "quality_grade": quality_grade,
            "place_guess": place_guess,
            "latitude": self._parse_float(latitude_raw),
            "longitude": self._parse_float(longitude_raw),
            "latitude_raw": latitude_raw,
            "longitude_raw": longitude_raw,
            "geoprivacy": geoprivacy,
            "url": url,
            "description": description,
            "tags": tags,
            "source_file": str(path),
            "row": dict(row),
        }
        return KnowledgeUnit(
            source_project=SourceProject.INATURALIST_OBSERVATIONS_CSV,
            source_id=self._source_id(observation_id, row),
            source_entity_type="observation",
            title=self._title(common_name, scientific_name, observed_at),
            content=self._content(common_name, scientific_name, quality_grade, place_guess, description, url),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=self._dedupe(["inaturalist", iconic_taxon.lower() if iconic_taxon else "", quality_grade.lower() if quality_grade else "", *tags]),
            created_at=observed_at or created_at or comparable_at,
            updated_at=comparable_at,
        )

    def _source_id(self, observation_id: str, row: dict[str, Any]) -> str:
        if observation_id:
            return f"inaturalist_observations_csv:{observation_id}"
        raw = "|".join(f"{key}={row[key]}" for key in sorted(row))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"inaturalist_observations_csv:{digest}"

    def _title(self, common_name: str, scientific_name: str, observed_at: datetime | None) -> str:
        name = common_name or scientific_name or "iNaturalist observation"
        if observed_at:
            return f"{name} on {observed_at.date().isoformat()}"
        return name

    def _content(
        self,
        common_name: str,
        scientific_name: str,
        quality_grade: str,
        place_guess: str,
        description: str,
        url: str,
    ) -> str:
        parts: list[str] = []
        if common_name:
            parts.append(f"Common name: {common_name}")
        if scientific_name:
            parts.append(f"Scientific name: {scientific_name}")
        if quality_grade:
            parts.append(f"Quality grade: {quality_grade}")
        if place_guess:
            parts.append(f"Place: {place_guess}")
        if url:
            parts.append(f"URL: {url}")
        if description:
            parts.append(f"\nDescription:\n{description}")
        return "\n".join(parts)

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).lower(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.lower())
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _split_tags(self, value: str) -> list[str]:
        return self._dedupe(part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip())

    def _dedupe(self, values: Any) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in result:
                result.append(text)
        return result

    def _parse_float(self, value: str) -> float | str | None:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return value

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        text = value.strip()
        for candidate in (text, f"{text}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                pass
        for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
