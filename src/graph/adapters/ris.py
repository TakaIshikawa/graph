"""Adapter for local RIS reference files."""

from __future__ import annotations

import hashlib
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


TITLE_TAGS = ("TI", "T1")
AUTHOR_TAGS = ("AU", "A1")
ABSTRACT_TAGS = ("AB", "N2")
DATE_TAGS = ("DA", "Y1", "PY")
KEYWORD_TAGS = ("KW",)
VENUE_TAGS = ("T2", "JF", "JO", "J1", "JA", "PB", "CY")


class RisAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "ris"

    @property
    def entity_types(self) -> list[str]:
        return ["ris_record"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "ris_record" not in entity_types:
            return result

        sync_at = self._sync_datetime(since) if since else None
        root = Path(self.path).expanduser()
        root = root if root.is_dir() else root.parent
        malformed_records = 0

        for path in self._discover_paths():
            try:
                text = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError):
                malformed_records += 1
                continue

            records, malformed = self._parse_records(text)
            malformed_records += malformed
            for record in records:
                source_date = self._record_date(record)
                if sync_at and source_date and source_date <= sync_at:
                    continue

                unit = self._unit_from_record(root, path, record)
                if unit is None:
                    malformed_records += 1
                    continue
                result.units.append(unit)

        if malformed_records:
            suffix = "s" if malformed_records != 1 else ""
            warnings.warn(
                f"Skipped {malformed_records} malformed RIS record{suffix}.",
                stacklevel=2,
            )

        return result

    def _discover_paths(self) -> list[Path]:
        sources = [
            source.strip()
            for source in re.split(r"[\n,]", self.path)
            if source.strip()
        ]
        paths: list[Path] = []
        for source in sources:
            configured = Path(source).expanduser()
            if configured.is_file() and configured.suffix.lower() == ".ris":
                paths.append(configured)
            elif configured.is_dir():
                paths.extend(
                    item
                    for item in sorted(configured.rglob("*"))
                    if item.is_file() and item.suffix.lower() == ".ris"
                )
        return paths

    def _parse_records(self, text: str) -> tuple[list[dict[str, list[str]]], int]:
        records: list[dict[str, list[str]]] = []
        malformed = 0
        current: dict[str, list[str]] | None = None
        current_tag = ""
        saw_er = False

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                continue

            match = re.match(r"^([A-Z0-9]{2})\s{2}-\s?(.*)$", line)
            if match is None:
                if current is not None and current_tag:
                    current[current_tag][-1] = f"{current[current_tag][-1]} {line.strip()}".strip()
                continue

            tag, value = match.group(1), self._clean_text(match.group(2))
            if tag == "TY":
                if current is not None and not saw_er:
                    malformed += 1
                current = {"TY": [value]}
                current_tag = "TY"
                saw_er = False
                continue

            if current is None:
                malformed += 1
                continue

            current.setdefault(tag, []).append(value)
            current_tag = tag
            if tag == "ER":
                if self._is_complete_record(current):
                    records.append(current)
                else:
                    malformed += 1
                current = None
                current_tag = ""
                saw_er = True

        if current is not None:
            malformed += 1

        return records, malformed

    def _unit_from_record(
        self,
        root: Path,
        path: Path,
        record: dict[str, list[str]],
    ) -> KnowledgeUnit | None:
        title = self._first(record, TITLE_TAGS)
        content = self._content(record) or title
        if not title:
            return None

        source_file = path.relative_to(root).as_posix()
        date = self._record_date(record)
        metadata = {
            "ris_type": self._first(record, ("TY",)),
            "authors": self._all(record, AUTHOR_TAGS),
            "year": self._year(record),
            "date": self._first(record, DATE_TAGS),
            "doi": self._doi(record),
            "url": self._url(record),
            "venue": self._first(record, VENUE_TAGS),
            "source_file": source_file,
        }

        unit = KnowledgeUnit(
            source_project=SourceProject.RIS,
            source_id=self._source_id(record),
            source_entity_type="ris_record",
            title=title,
            content=content,
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=self._tags(record),
            created_at=date or datetime.now(timezone.utc),
        )
        if date is not None:
            unit.updated_at = date
        return unit

    def _content(self, record: dict[str, list[str]]) -> str:
        parts: list[str] = []
        authors = self._all(record, AUTHOR_TAGS)
        year = self._year(record)
        venue = self._first(record, VENUE_TAGS)
        abstract = self._first(record, ABSTRACT_TAGS)
        doi = self._doi(record)
        url = self._url(record)

        if authors:
            parts.append(f"Authors: {'; '.join(authors)}")
        if year:
            parts.append(f"Year: {year}")
        if venue:
            parts.append(f"Venue: {venue}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        if doi:
            parts.append(f"DOI: {doi}")
        if url:
            parts.append(f"URL: {url}")
        return "\n\n".join(parts)

    def _source_id(self, record: dict[str, list[str]]) -> str:
        doi = self._doi(record).lower()
        if doi:
            return f"doi:{doi}"
        url = self._url(record)
        if url:
            return f"url:{url}"
        stable_parts = []
        for tag in sorted(record):
            if tag == "ER":
                continue
            stable_parts.extend(f"{tag}:{value}" for value in record[tag])
        digest = hashlib.sha256("\n".join(stable_parts).encode("utf-8")).hexdigest()
        return f"ris:{digest[:24]}"

    def _is_complete_record(self, record: dict[str, list[str]]) -> bool:
        return bool(self._first(record, ("TY",)) and self._first(record, TITLE_TAGS))

    def _first(self, record: dict[str, list[str]], tags: tuple[str, ...]) -> str:
        for tag in tags:
            for value in record.get(tag, []):
                if value:
                    return value
        return ""

    def _all(self, record: dict[str, list[str]], tags: tuple[str, ...]) -> list[str]:
        values: list[str] = []
        for tag in tags:
            for value in record.get(tag, []):
                if value and value not in values:
                    values.append(value)
        return values

    def _doi(self, record: dict[str, list[str]]) -> str:
        doi = self._first(record, ("DO",))
        return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE).strip()

    def _url(self, record: dict[str, list[str]]) -> str:
        return self._first(record, ("UR", "L2"))

    def _year(self, record: dict[str, list[str]]) -> str:
        for tag in DATE_TAGS:
            for value in record.get(tag, []):
                match = re.search(r"\b(\d{4})\b", value)
                if match:
                    return match.group(1)
        return ""

    def _tags(self, record: dict[str, list[str]]) -> list[str]:
        tags: list[str] = []
        for value in self._all(record, KEYWORD_TAGS):
            for raw_tag in re.split(r"[,;]", value):
                tag = raw_tag.strip().removeprefix("#").strip()
                if tag and tag not in tags:
                    tags.append(tag)
        return tags

    def _record_date(self, record: dict[str, list[str]]) -> datetime | None:
        for tag in DATE_TAGS:
            for value in record.get(tag, []):
                parsed = self._parse_datetime(value)
                if parsed is not None:
                    return parsed
        return None

    def _parse_datetime(self, value: str) -> datetime | None:
        value = value.strip()
        if not value:
            return None

        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            match = re.search(r"\b(\d{4})(?:[-/](\d{1,2}))?(?:[-/](\d{1,2}))?", value)
            if match is None:
                return None
            year = int(match.group(1))
            month = int(match.group(2) or "1")
            day = int(match.group(3) or "1")
            try:
                parsed = datetime(year, month, day)
            except ValueError:
                return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _clean_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()
