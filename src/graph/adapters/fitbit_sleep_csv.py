"""Adapter for Fitbit sleep CSV exports."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class FitbitSleepCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "fitbit_sleep_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["sleep"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "sleep" not in entity_types:
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.created_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.created_at, unit.source_id)))
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
            return [dict(row) for row in csv.DictReader(handle)]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        start = self._parse_datetime(self._first(row, "Start Time", "Start", "Sleep Start", "startTime"))
        date = self._parse_datetime(self._first(row, "Date", "Sleep Date"))
        created_at = start or date
        if created_at is None:
            return None

        end = self._parse_datetime(self._first(row, "End Time", "End", "Sleep End", "endTime"))
        score = self._parse_int(self._first(row, "Sleep Score", "Score", "Overall Score"))
        metadata = {
            "start_time": start.isoformat() if start else created_at.isoformat(),
            "end_time": end.isoformat() if end else None,
            "minutes_asleep": self._parse_int(self._first(row, "Minutes Asleep", "Minutes Sleeping", "Asleep")),
            "minutes_awake": self._parse_int(self._first(row, "Minutes Awake", "Awake")),
            "awakenings": self._parse_int(self._first(row, "Number of Awakenings", "Awakenings", "Times Awakened")),
            "time_in_bed": self._parse_int(self._first(row, "Time in Bed", "Minutes in Bed")),
            "sleep_score": score,
            "deep_minutes": self._parse_int(self._first(row, "Deep Sleep", "Deep")),
            "light_minutes": self._parse_int(self._first(row, "Light Sleep", "Light")),
            "rem_minutes": self._parse_int(self._first(row, "REM Sleep", "REM")),
            "wake_minutes": self._parse_int(self._first(row, "Wake", "Wake Sleep")),
            "source_file": source_file,
        }
        tags = ["fitbit", "sleep"]
        band = self._score_band(score)
        if band:
            tags.append(band)

        return KnowledgeUnit(
            source_project=SourceProject.FITBIT_SLEEP_CSV,
            source_id=self._source_id(row, created_at, end),
            source_entity_type="sleep",
            title=f"Fitbit sleep {created_at.date().isoformat()}",
            content=self._content(created_at, end, metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=tags,
            created_at=created_at,
            updated_at=end or created_at,
        )

    def _source_id(self, row: dict[str, Any], start: datetime, end: datetime | None) -> str:
        explicit = self._first(row, "ID", "Log ID")
        raw = explicit or "|".join([start.isoformat(), end.isoformat() if end else "", self._first(row, "Sleep Score", "Score")])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"fitbit_sleep_csv:{digest}"

    def _score_band(self, score: int | None) -> str:
        if score is None:
            return ""
        if score >= 90:
            return "excellent_sleep"
        if score >= 80:
            return "good_sleep"
        if score >= 60:
            return "fair_sleep"
        return "poor_sleep"

    def _content(self, start: datetime, end: datetime | None, metadata: dict[str, Any]) -> str:
        parts = [f"Start: {start.isoformat()}"]
        if end:
            parts.append(f"End: {end.isoformat()}")
        if metadata.get("minutes_asleep") is not None:
            parts.append(f"Minutes asleep: {metadata['minutes_asleep']}")
        if metadata.get("sleep_score") is not None:
            parts.append(f"Sleep score: {metadata['sleep_score']}")
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

    def _parse_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(float(str(value).strip()))
        except ValueError:
            return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        text = str(value).strip()
        for candidate in (text, f"{text}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                pass
        for fmt in ("%m/%d/%Y", "%m/%d/%y", "%m/%d/%Y %I:%M %p", "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
