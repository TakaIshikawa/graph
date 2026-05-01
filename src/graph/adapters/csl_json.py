"""Adapter for CSL-JSON bibliography exports."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


DATE_FIELDS = ("issued", "event-date", "submitted", "accessed")


class CslJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "csl_json"

    @property
    def entity_types(self) -> list[str]:
        return ["csl_json_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "csl_json_item" not in entity_types:
            return result

        paths = self._discover_paths()
        if not paths:
            return result

        sync_at = self._sync_timestamp(since) if since else None
        root = Path(self.path).expanduser()
        root = root if root.is_dir() else root.parent
        malformed_files = 0
        malformed_items = 0

        for path in paths:
            stat = path.stat()
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                malformed_files += 1
                continue

            items = self._items(data)
            if items is None:
                malformed_files += 1
                continue

            for index, item in enumerate(items):
                unit = self._unit_from_item(
                    root,
                    path,
                    item,
                    index=index,
                    created_timestamp=stat.st_ctime,
                )
                if unit is None:
                    malformed_items += 1
                    continue
                result.units.append(unit)

        skipped = malformed_files + malformed_items
        if skipped:
            suffix = "s" if skipped != 1 else ""
            warnings.warn(
                f"Skipped {skipped} malformed CSL-JSON input{suffix}.",
                stacklevel=2,
            )

        return result

    def _discover_paths(self) -> list[Path]:
        configured = Path(self.path).expanduser()
        if configured.is_file() and configured.suffix.lower() == ".json":
            return [configured]
        if configured.is_dir():
            return sorted(
                item
                for item in configured.rglob("*")
                if item.is_file() and item.suffix.lower() == ".json"
            )
        return []

    def _items(self, data: Any) -> list[dict[str, Any]] | None:
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data
        return None

    def _unit_from_item(
        self,
        root: Path,
        path: Path,
        item: dict[str, Any],
        *,
        index: int,
        created_timestamp: float,
    ) -> KnowledgeUnit | None:
        title = self._string(item.get("title")) or self._string(item.get("shortTitle"))
        if not title:
            return None

        source_file = path.relative_to(root).as_posix()
        issued = self._date(item)
        authors = self._authors(item.get("author"))
        container_title = self._string(item.get("container-title"))
        publisher = self._string(item.get("publisher"))
        doi = self._doi(item)
        url = self._string(item.get("URL")) or self._string(item.get("url"))
        metadata = {
            "csl_type": self._string(item.get("type")),
            "doi": doi,
            "url": url,
            "issued": self._date_text(item),
            "authors": authors,
            "publisher": publisher,
            "container_title": container_title,
            "source_file": source_file,
        }

        unit = KnowledgeUnit(
            source_project=SourceProject.CSL_JSON,
            source_id=self._source_id(source_file, item, index),
            source_entity_type="csl_json_item",
            title=title,
            content=self._content(item, authors, container_title or publisher, doi, url),
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=self._tags(item),
            created_at=issued or datetime.fromtimestamp(created_timestamp, tz=timezone.utc),
        )
        if issued is not None:
            unit.updated_at = issued
        return unit

    def _content(
        self,
        item: dict[str, Any],
        authors: list[str],
        venue: str,
        doi: str,
        url: str,
    ) -> str:
        parts: list[str] = []
        issued = self._date_text(item)
        abstract = self._string(item.get("abstract"))
        if authors:
            parts.append(f"Authors: {'; '.join(authors)}")
        if issued:
            parts.append(f"Issued: {issued}")
        if venue:
            parts.append(f"Venue: {venue}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        if doi:
            parts.append(f"DOI: {doi}")
        if url:
            parts.append(f"URL: {url}")
        return "\n\n".join(parts)

    def _source_id(self, source_file: str, item: dict[str, Any], index: int) -> str:
        item_id = self._string(item.get("id"))
        if item_id:
            return item_id

        doi = self._doi(item).lower()
        if doi:
            return f"doi:{doi}"

        url = self._string(item.get("URL")) or self._string(item.get("url"))
        if url:
            return f"url:{url}"

        stable = json.dumps(item, sort_keys=True, ensure_ascii=True, default=str)
        digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()
        return f"{source_file}:{index}:{digest[:24]}"

    def _authors(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []

        authors: list[str] = []
        for person in value:
            if not isinstance(person, dict):
                continue
            name = self._string(person.get("literal"))
            if not name:
                given = self._string(person.get("given"))
                family = self._string(person.get("family"))
                if given and family:
                    name = f"{family}, {given}"
                else:
                    name = family or given
            if name and name not in authors:
                authors.append(name)
        return authors

    def _date(self, item: dict[str, Any]) -> datetime | None:
        for field in DATE_FIELDS:
            parsed = self._parse_date(item.get(field))
            if parsed is not None:
                return parsed
        return None

    def _date_text(self, item: dict[str, Any]) -> str:
        for field in DATE_FIELDS:
            value = item.get(field)
            text = self._date_value_text(value)
            if text:
                return text
        return ""

    def _parse_date(self, value: Any) -> datetime | None:
        parts = self._date_parts(value)
        if not parts:
            return None

        year = parts[0]
        month = parts[1] if len(parts) > 1 else 1
        day = parts[2] if len(parts) > 2 else 1
        try:
            return datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            return None

    def _date_value_text(self, value: Any) -> str:
        if isinstance(value, dict):
            literal = self._string(value.get("literal"))
            if literal:
                return literal
            raw = self._string(value.get("raw"))
            if raw:
                return raw

        parts = self._date_parts(value)
        if not parts:
            return ""
        return "-".join(f"{part:02d}" if index else str(part) for index, part in enumerate(parts))

    def _date_parts(self, value: Any) -> list[int]:
        if not isinstance(value, dict):
            return []
        date_parts = value.get("date-parts")
        if not isinstance(date_parts, list) or not date_parts:
            return []
        first = date_parts[0]
        if not isinstance(first, list):
            return []

        parts: list[int] = []
        for raw in first[:3]:
            if isinstance(raw, int):
                parts.append(raw)
            elif isinstance(raw, str) and raw.isdigit():
                parts.append(int(raw))
            else:
                break
        return parts

    def _tags(self, item: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        for key in ("keyword", "keywords", "categories"):
            value = item.get(key)
            if isinstance(value, list):
                candidates = value
            else:
                candidates = re.split(r"[,;]", self._string(value))
            for raw_tag in candidates:
                tag = self._string(raw_tag).removeprefix("#").strip()
                if tag and tag not in tags:
                    tags.append(tag)

        item_type = self._string(item.get("type"))
        if item_type and item_type not in tags:
            tags.append(item_type)
        return tags

    def _doi(self, item: dict[str, Any]) -> str:
        doi = self._string(item.get("DOI")) or self._string(item.get("doi"))
        return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE).strip()

    def _string(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value).strip()
        if isinstance(value, (int, float)):
            return str(value)
        return ""

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()
