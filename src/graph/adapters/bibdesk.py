"""Adapter for BibDesk plist bibliography exports."""

from __future__ import annotations

import hashlib
import json
import plistlib
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


TITLE_FIELDS = ("title", "Title")
AUTHOR_FIELDS = ("authors", "Authors", "author", "Author")
ABSTRACT_FIELDS = ("abstract", "Abstract")
VENUE_FIELDS = (
    "journal",
    "Journal",
    "booktitle",
    "Booktitle",
    "BookTitle",
    "container-title",
    "Publication",
)
YEAR_FIELDS = ("year", "Year", "date", "Date")
DOI_FIELDS = ("doi", "DOI", "Doi")
URL_FIELDS = ("url", "URL", "Url")
CITE_KEY_FIELDS = (
    "cite_key",
    "citation_key",
    "citekey",
    "citeKey",
    "Cite Key",
    "BibTeX Cite Key",
    "Key",
    "key",
)
TYPE_FIELDS = ("type", "Type", "publication_type", "BibTeX Type", "pubType")


class BibDeskAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "bibdesk"

    @property
    def entity_types(self) -> list[str]:
        return ["publication"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "publication" not in entity_types:
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
                with path.open("rb") as handle:
                    data = plistlib.load(handle)
            except (OSError, plistlib.InvalidFileException, ValueError):
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
                f"Skipped {skipped} malformed BibDesk input{suffix}.",
                stacklevel=2,
            )

        return result

    def _discover_paths(self) -> list[Path]:
        configured = Path(self.path).expanduser()
        if configured.is_file() and configured.suffix.lower() == ".plist":
            return [configured]
        if configured.is_dir():
            return sorted(
                item
                for item in configured.rglob("*")
                if item.is_file() and item.suffix.lower() == ".plist"
            )
        return []

    def _items(self, data: Any) -> list[dict[str, Any]] | None:
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data
        if not isinstance(data, dict):
            return None

        for key in ("publications", "Publications", "items", "Items"):
            value = data.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
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
        title = self._first(item, TITLE_FIELDS)
        if not title:
            return None

        source_file = path.relative_to(root).as_posix()
        authors = self._authors(self._first_value(item, AUTHOR_FIELDS))
        year = self._year(self._first(item, YEAR_FIELDS))
        doi = self._first(item, DOI_FIELDS)
        url = self._first(item, URL_FIELDS)
        publication_type = self._first(item, TYPE_FIELDS)
        cite_key = self._first(item, CITE_KEY_FIELDS)
        venue = self._first(item, VENUE_FIELDS)
        abstract = self._first(item, ABSTRACT_FIELDS)
        metadata = {
            "cite_key": cite_key,
            "doi": doi,
            "url": url,
            "publication_type": publication_type,
            "authors": authors,
            "year": year,
            "source_file": source_file,
            "raw_keys": sorted(str(key) for key in item),
        }

        return KnowledgeUnit(
            source_project=SourceProject.BIBDESK,
            source_id=self._source_id(source_file, item, index, cite_key, doi, url),
            source_entity_type="publication",
            title=title,
            content=self._content(title, authors, abstract, venue, year, doi, url),
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            created_at=datetime.fromtimestamp(created_timestamp, tz=timezone.utc),
        )

    def _content(
        self,
        title: str,
        authors: list[str],
        abstract: str,
        venue: str,
        year: str,
        doi: str,
        url: str,
    ) -> str:
        parts = [f"Title: {title}"]
        if authors:
            parts.append(f"Authors: {'; '.join(authors)}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        if venue:
            parts.append(f"Venue: {venue}")
        if year:
            parts.append(f"Year: {year}")
        if doi:
            parts.append(f"DOI: {doi}")
        if url:
            parts.append(f"URL: {url}")
        return "\n\n".join(parts)

    def _source_id(
        self,
        source_file: str,
        item: dict[str, Any],
        index: int,
        cite_key: str,
        doi: str,
        url: str,
    ) -> str:
        if cite_key:
            return f"{source_file}:{cite_key}"
        if doi:
            return f"doi:{doi.lower()}"
        if url:
            return f"url:{url}"

        stable = json.dumps(item, sort_keys=True, ensure_ascii=True, default=str)
        digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()
        return f"{source_file}:{index}:{digest[:24]}"

    def _authors(self, value: Any) -> list[str]:
        if isinstance(value, list):
            authors = [self._string(item) for item in value]
        else:
            authors = [
                author.strip()
                for author in re.split(r"\s+\band\b\s+|;", self._string(value))
            ]
        return [author for author in authors if author]

    def _year(self, value: str) -> str:
        match = re.search(r"\b(\d{4})\b", value)
        return match.group(1) if match else value

    def _first(self, item: dict[str, Any], keys: tuple[str, ...]) -> str:
        return self._string(self._first_value(item, keys))

    def _first_value(self, item: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            value = item.get(key)
            if self._string(value) or isinstance(value, list):
                return value
        return ""

    def _string(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").strip()
        if isinstance(value, (list, dict)):
            return ""
        return str(value).strip()

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()
