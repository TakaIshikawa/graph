"""Adapter for Crossref works JSON exports."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class CrossrefAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "crossref"

    @property
    def entity_types(self) -> list[str]:
        return ["work"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "work" not in entity_types:
            return result

        paths = self._discover_paths()
        if not paths:
            return result

        sync_at = self._sync_timestamp(since) if since else None
        configured = Path(self.path).expanduser()
        root = configured if configured.is_dir() else configured.parent
        malformed_files = 0
        malformed_items = 0
        reference_dois_by_source_id: dict[str, list[str]] = {}

        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                malformed_files += 1
                continue
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
                reference_dois_by_source_id[unit.source_id] = self._reference_dois(item)

        doi_index = {
            self._normalize_doi(unit.metadata.get("doi")): unit.source_id
            for unit in result.units
            if self._normalize_doi(unit.metadata.get("doi"))
        }
        emitted_edges: set[tuple[str, str]] = set()
        for source_id, reference_dois in reference_dois_by_source_id.items():
            for doi in reference_dois:
                target_id = doi_index.get(doi)
                if not target_id or target_id == source_id:
                    continue
                edge_key = (source_id, target_id)
                if edge_key in emitted_edges:
                    continue
                emitted_edges.add(edge_key)
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(source_id, target_id),
                        from_unit_id=source_id,
                        to_unit_id=target_id,
                        relation=EdgeRelation.REFERENCES,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.CROSSREF.value,
                            "from_entity_type": "work",
                            "to_entity_type": "work",
                            "relation_type": "crossref_reference",
                            "doi": doi,
                        },
                    )
                )

        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id))

        skipped = malformed_files + malformed_items
        if skipped:
            suffix = "s" if skipped != 1 else ""
            warnings.warn(
                f"Skipped {skipped} malformed Crossref input{suffix}.",
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
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data
        if not isinstance(data, dict):
            return None

        message = data.get("message")
        if isinstance(message, dict):
            items = message.get("items")
            if isinstance(items, list) and all(isinstance(item, dict) for item in items):
                return items
        if self._string(data.get("DOI")) or self._string(data.get("doi")):
            return [data]
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
        title = self._first_text(item.get("title"))
        doi = self._normalize_doi(item.get("DOI") or item.get("doi"))
        if not title:
            title = doi or self._string(item.get("URL")) or self._string(item.get("url"))
        if not title:
            return None

        source_file = path.relative_to(root).as_posix()
        issued = self._parse_date(item.get("issued")) or self._parse_date(item.get("published-print"))
        issued_text = self._date_value_text(item.get("issued")) or self._date_value_text(item.get("published-print"))
        authors = self._authors(item.get("author"))
        container_titles = self._text_list(item.get("container-title"))
        subjects = self._text_list(item.get("subject"))
        publisher = self._string(item.get("publisher"))
        url = self._string(item.get("URL")) or self._string(item.get("url"))
        abstract = self._clean_markup(self._string(item.get("abstract")))
        source_id = self._source_id(source_file, item, index, doi)

        metadata = {
            "doi": doi,
            "url": url,
            "type": self._string(item.get("type")),
            "authors": authors,
            "publisher": publisher,
            "issued": issued_text,
            "container_title": container_titles[0] if container_titles else "",
            "container_titles": container_titles,
            "subjects": subjects,
            "reference_dois": self._reference_dois(item),
            "source_file": source_file,
        }

        unit = KnowledgeUnit(
            source_project=SourceProject.CROSSREF,
            source_id=source_id,
            source_entity_type="work",
            title=title,
            content=self._content(authors, issued_text, container_titles, publisher, abstract, doi, url),
            content_type=ContentType.FINDING,
            metadata=metadata,
            tags=self._tags(subjects, container_titles),
            created_at=issued or datetime.fromtimestamp(created_timestamp, tz=timezone.utc),
        )
        if issued is not None:
            unit.updated_at = issued
        return unit

    def _content(
        self,
        authors: list[str],
        issued: str,
        container_titles: list[str],
        publisher: str,
        abstract: str,
        doi: str,
        url: str,
    ) -> str:
        parts: list[str] = []
        if authors:
            parts.append(f"Authors: {'; '.join(authors)}")
        if issued:
            parts.append(f"Issued: {issued}")
        venue = container_titles[0] if container_titles else publisher
        if venue:
            parts.append(f"Venue: {venue}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        if doi:
            parts.append(f"DOI: {doi}")
        if url:
            parts.append(f"URL: {url}")
        return "\n\n".join(parts)

    def _source_id(self, source_file: str, item: dict[str, Any], index: int, doi: str) -> str:
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
            name = self._string(person.get("name"))
            if not name:
                given = self._string(person.get("given"))
                family = self._string(person.get("family"))
                if given and family:
                    name = f"{given} {family}"
                else:
                    name = family or given
            if name and name not in authors:
                authors.append(name)
        return authors

    def _reference_dois(self, item: dict[str, Any]) -> list[str]:
        references = item.get("reference")
        if not isinstance(references, list):
            return []

        dois: list[str] = []
        for reference in references:
            if not isinstance(reference, dict):
                continue
            doi = self._normalize_doi(reference.get("DOI") or reference.get("doi"))
            if doi and doi not in dois:
                dois.append(doi)
        return dois

    def _tags(self, subjects: list[str], container_titles: list[str]) -> list[str]:
        tags: list[str] = []
        for value in [*subjects, *container_titles]:
            tag = value.strip()
            if tag and tag not in tags:
                tags.append(tag)
        return tags

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

    def _first_text(self, value: Any) -> str:
        values = self._text_list(value)
        return values[0] if values else self._string(value)

    def _text_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            texts = [self._string(item) for item in value]
            return [text for text in texts if text]
        text = self._string(value)
        return [text] if text else []

    def _normalize_doi(self, value: Any) -> str:
        doi = self._string(value)
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
        return doi.strip().lower()

    def _clean_markup(self, value: str) -> str:
        return re.sub(r"<[^>]+>", " ", value).strip()

    def _string(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value).strip()
        if isinstance(value, (int, float)):
            return str(value)
        return ""

    def _sync_timestamp(self, since: SyncState) -> float:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).timestamp()

    def _edge_id(self, source_id: str, target_id: str) -> str:
        digest = hashlib.sha256(f"{source_id}\0{target_id}".encode("utf-8")).hexdigest()
        return f"crossref-ref:{digest[:24]}"
