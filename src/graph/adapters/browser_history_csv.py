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
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class BrowserHistoryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "browser_history_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["web_history", "domain", "visit_hour"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types) if entity_types is not None else {"web_history", "domain"}
        if not allowed_types.intersection(self.entity_types):
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

        web_history_units = sorted(units.values(), key=lambda unit: unit.source_id)
        domain_units = self._domain_units(web_history_units) if "domain" in allowed_types else []
        visit_hour_units = self._visit_hour_units(web_history_units) if "visit_hour" in allowed_types else []

        if "web_history" in allowed_types:
            result.units.extend(web_history_units)
        if "domain" in allowed_types:
            result.units.extend(domain_units)
        if "visit_hour" in allowed_types:
            result.units.extend(visit_hour_units)
        if {"web_history", "domain"}.issubset(allowed_types):
            result.edges.extend(self._domain_edges(web_history_units, domain_units))
        if {"web_history", "visit_hour"}.issubset(allowed_types):
            result.edges.extend(self._visit_hour_edges(web_history_units, visit_hour_units))

        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
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
        referrer_url = self._first(row, "referrer", "referrer_url", "from_url", "source_url")
        metadata = {
            "url": url,
            "normalized_url": normalized_url,
            "domain": domain,
            "visit_time": visit_time_text,
            "last_visit_time": last_visit_time_text,
            "visit_timestamps": self._visit_timestamps(visit_at, last_visit_at),
            "visit_count": self._parse_int(self._first(row, "visit_count", "visits")),
            "typed_count": self._parse_int(self._first(row, "typed_count", "typed")),
            "source_file": source_file,
        }
        if referrer_url:
            metadata["referrer_url"] = referrer_url

        return KnowledgeUnit(
            source_project=SourceProject.BROWSER_HISTORY_CSV,
            source_id=self._source_id(normalized_url),
            source_entity_type="web_history",
            title=title,
            content=self._content(title, normalized_url),
            content_type=ContentType.METADATA,
            metadata=metadata,
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
        if not hostname or any(char.isspace() for char in hostname):
            return ""
        try:
            port = parsed.port
        except ValueError:
            return ""
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

    def _domain_units(self, units: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for unit in units:
            domain = str(unit.metadata.get("domain") or "").strip().lower()
            if domain:
                grouped.setdefault(domain, []).append(unit)

        domain_units: list[KnowledgeUnit] = []
        for domain, visits in sorted(grouped.items()):
            ordered = sorted(visits, key=lambda unit: unit.source_id)
            created_at = min(unit.created_at for unit in ordered)
            updated_at = max(unit.updated_at for unit in ordered)
            domain_units.append(
                KnowledgeUnit(
                    source_project=SourceProject.BROWSER_HISTORY_CSV,
                    source_id=self._domain_source_id(domain),
                    source_entity_type="domain",
                    title=domain,
                    content=f"Browser history domain: {domain}\nVisits: {len(ordered)}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "domain": domain,
                        "visit_count": len(ordered),
                        "page_source_ids": [unit.source_id for unit in ordered],
                        "normalized_urls": sorted(
                            str(unit.metadata.get("normalized_url"))
                            for unit in ordered
                            if unit.metadata.get("normalized_url")
                        ),
                        "source_files": sorted(
                            {
                                str(unit.metadata.get("source_file"))
                                for unit in ordered
                                if unit.metadata.get("source_file")
                            }
                        ),
                    },
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return domain_units

    def _domain_edges(
        self,
        web_history_units: list[KnowledgeUnit],
        domain_units: list[KnowledgeUnit],
    ) -> list[KnowledgeEdge]:
        domain_source_ids = {
            str(unit.metadata.get("domain") or ""): unit.source_id
            for unit in domain_units
            if unit.metadata.get("domain")
        }
        edges: list[KnowledgeEdge] = []
        for unit in web_history_units:
            domain = str(unit.metadata.get("domain") or "")
            domain_source_id = domain_source_ids.get(domain)
            if not domain_source_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=self._domain_edge_id(unit.source_id, domain_source_id),
                    from_unit_id=unit.source_id,
                    to_unit_id=domain_source_id,
                    relation=EdgeRelation.RELATES_TO,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.BROWSER_HISTORY_CSV.value,
                        "from_entity_type": "web_history",
                        "to_entity_type": "domain",
                        "relation_type": "visit_domain",
                        "domain": domain,
                    },
                    created_at=unit.created_at,
                )
            )
        return edges

    def _domain_source_id(self, domain: str) -> str:
        digest = hashlib.sha256(domain.encode("utf-8")).hexdigest()[:24]
        return f"browser_history_csv:domain:{digest}"

    def _domain_edge_id(self, visit_source_id: str, domain_source_id: str) -> str:
        raw = "|".join([SourceProject.BROWSER_HISTORY_CSV.value, visit_source_id, domain_source_id])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"browser-history-csv-domain-{digest}"

    def _visit_hour_units(self, units: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for unit in units:
            for hour in self._visit_hours(unit):
                grouped.setdefault(hour, []).append(unit)

        hour_units: list[KnowledgeUnit] = []
        for hour, visits in sorted(grouped.items()):
            ordered = sorted({unit.source_id: unit for unit in visits}.values(), key=lambda unit: unit.source_id)
            parsed_hour = self._parse_datetime(hour)
            created_at = parsed_hour or min(unit.created_at for unit in ordered)
            updated_at = max(unit.updated_at for unit in ordered)
            hour_units.append(
                KnowledgeUnit(
                    source_project=SourceProject.BROWSER_HISTORY_CSV,
                    source_id=self._visit_hour_source_id(hour),
                    source_entity_type="visit_hour",
                    title=hour,
                    content=f"Browser history visit hour: {hour}\nVisits: {len(ordered)}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "visit_hour": hour,
                        "visit_count": len(ordered),
                        "page_source_ids": [unit.source_id for unit in ordered],
                        "domains": sorted({str(unit.metadata.get("domain")) for unit in ordered if unit.metadata.get("domain")}),
                        "source_files": sorted({str(unit.metadata.get("source_file")) for unit in ordered if unit.metadata.get("source_file")}),
                    },
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return hour_units

    def _visit_hour_edges(
        self,
        web_history_units: list[KnowledgeUnit],
        visit_hour_units: list[KnowledgeUnit],
    ) -> list[KnowledgeEdge]:
        hour_source_ids = {str(unit.metadata.get("visit_hour")): unit.source_id for unit in visit_hour_units}
        edges: list[KnowledgeEdge] = []
        for unit in web_history_units:
            for hour in self._visit_hours(unit):
                hour_source_id = hour_source_ids.get(hour)
                if not hour_source_id:
                    continue
                edges.append(
                    KnowledgeEdge(
                        id=self._visit_hour_edge_id(unit.source_id, hour_source_id),
                        from_unit_id=unit.source_id,
                        to_unit_id=hour_source_id,
                        relation=EdgeRelation.RELATES_TO,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.BROWSER_HISTORY_CSV.value,
                            "from_entity_type": "web_history",
                            "to_entity_type": "visit_hour",
                            "relation_type": "visit_hour",
                            "visit_hour": hour,
                        },
                        created_at=unit.created_at,
                    )
                )
        return edges

    def _visit_hours(self, unit: KnowledgeUnit) -> list[str]:
        hours: list[str] = []
        for timestamp in unit.metadata.get("visit_timestamps") or []:
            parsed = self._parse_datetime(timestamp)
            if parsed is None:
                continue
            hour = parsed.replace(minute=0, second=0, microsecond=0).isoformat()
            if hour not in hours:
                hours.append(hour)
        return hours

    def _visit_hour_source_id(self, hour: str) -> str:
        digest = hashlib.sha256(hour.encode("utf-8")).hexdigest()[:24]
        return f"browser_history_csv:visit_hour:{digest}"

    def _visit_hour_edge_id(self, visit_source_id: str, hour_source_id: str) -> str:
        raw = "|".join([SourceProject.BROWSER_HISTORY_CSV.value, "visit_hour", visit_source_id, hour_source_id])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"browser-history-csv-visit-hour-{digest}"

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
