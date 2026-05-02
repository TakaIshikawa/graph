"""Adapter for browser history CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class BrowserHistoryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "browser_history_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["web_history"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "web_history" not in entity_types:
            return result

        sync_at = self._sync_datetime(since) if since else None
        units: dict[str, KnowledgeUnit] = {}
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue

            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                comparable_at = unit.updated_at or unit.created_at
                if sync_at and comparable_at <= sync_at:
                    continue
                units.setdefault(unit.source_id, unit)

        result.units.extend(sorted(units.values(), key=lambda unit: unit.source_id))
        return result

    def _iter_paths(self) -> list[Path]:
        paths: list[Path] = []
        for source in re.split(r"[\n,]", self.path):
            source = source.strip()
            if not source:
                continue
            path = Path(source).expanduser()
            if path.is_dir():
                paths.extend(sorted(child for child in path.rglob("*.csv") if child.is_file()))
            elif path.exists() and path.is_file():
                paths.append(path)
        return paths

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            rows: list[dict[str, Any]] = []
            for row in reader:
                normalized = {
                    self._canonical_key(str(key)): value
                    for key, value in row.items()
                    if key is not None
                }
                if normalized:
                    rows.append(normalized)
            return rows

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        url = self._first(row, "url", "uri", "link", "address")
        normalized_url = self._normalize_url(url)
        if not normalized_url:
            return None

        title = self._first(row, "title", "page_title", "name")
        if not title:
            title = self._title_from_url(normalized_url)

        visit_time_text = self._first(
            row,
            "visit_time",
            "visit_date",
            "visited_at",
            "time",
            "date",
            "timestamp",
        )
        last_visit_time_text = self._first(
            row,
            "last_visit_time",
            "last_visit",
            "last_visited",
            "last_visited_at",
            "last_visit_date",
            "date_last_visited",
        )
        visit_at = self._parse_datetime(visit_time_text)
        last_visit_at = self._parse_datetime(last_visit_time_text)
        created_at = visit_at or last_visit_at or datetime.now(timezone.utc)
        updated_at = last_visit_at or visit_at or created_at
        domain = urlsplit(normalized_url).hostname or ""

        return KnowledgeUnit(
            source_project=SourceProject.BROWSER_HISTORY_CSV,
            source_id=self._source_id(normalized_url),
            source_entity_type="web_history",
            title=title,
            content=self._content(title, normalized_url),
            content_type=ContentType.METADATA,
            metadata={
                "url": url,
                "normalized_url": normalized_url,
                "domain": domain,
                "visit_time": visit_time_text,
                "last_visit_time": last_visit_time_text,
                "visit_timestamps": self._visit_timestamps(visit_at, last_visit_at),
                "visit_count": self._parse_int(self._first(row, "visit_count", "visits")),
                "typed_count": self._parse_int(self._first(row, "typed_count", "typed")),
                "source_file": source_file,
            },
            created_at=created_at,
            updated_at=updated_at,
        )

    def _canonical_key(self, value: str) -> str:
        value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value.strip())
        value = re.sub(r"[^A-Za-z0-9]+", "_", value)
        return value.strip("_").lower()

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(self._canonical_key(key))
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _normalize_url(self, value: str) -> str:
        text = value.strip()
        if not text:
            return ""
        parsed = urlsplit(text)
        if not parsed.scheme and not parsed.netloc:
            parsed = urlsplit(f"https://{text}")
        if not parsed.netloc:
            return ""

        scheme = (parsed.scheme or "https").lower()
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return ""
        port = parsed.port
        netloc = hostname
        if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            netloc = f"{hostname}:{port}"
        path = parsed.path or "/"
        return urlunsplit((scheme, netloc, path, parsed.query, ""))

    def _title_from_url(self, normalized_url: str) -> str:
        parsed = urlsplit(normalized_url)
        suffix = parsed.path.strip("/")
        if suffix:
            return f"{parsed.hostname or normalized_url}/{suffix}"
        return parsed.hostname or normalized_url

    def _source_id(self, normalized_url: str) -> str:
        digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:24]
        return f"browser_history_csv:{digest}"

    def _content(self, title: str, normalized_url: str) -> str:
        return "\n".join([title, f"URL: {normalized_url}"])

    def _visit_timestamps(
        self, visit_at: datetime | None, last_visit_at: datetime | None
    ) -> list[str]:
        timestamps: list[str] = []
        for value in (visit_at, last_visit_at):
            if value is None:
                continue
            text = value.isoformat()
            if text not in timestamps:
                timestamps.append(text)
        return timestamps

    def _parse_int(self, value: str) -> int | None:
        if not value:
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return self._ensure_utc(value)
        if isinstance(value, int | float):
            return self._from_numeric_timestamp(float(value))

        text = str(value).strip()
        if not text:
            return None
        try:
            return self._from_numeric_timestamp(float(text))
        except ValueError:
            pass

        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return self._ensure_utc(parsed)

    def _from_numeric_timestamp(self, value: float) -> datetime | None:
        magnitude = abs(value)
        try:
            if magnitude >= 10_000_000_000_000_000:
                return datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=value)
            if magnitude >= 1_000_000_000_000_000:
                value /= 1_000_000
            elif magnitude >= 1_000_000_000_000:
                value /= 1_000
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    def _sync_datetime(self, since: SyncState) -> datetime:
        parsed = self._parse_datetime(since.last_sync_at)
        if parsed is None:
            raise ValueError(f"Invalid sync timestamp: {since.last_sync_at}")
        return parsed

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
