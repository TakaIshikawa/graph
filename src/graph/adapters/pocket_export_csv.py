"""Adapter for Pocket CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class PocketExportCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pocket_export_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["saved_item", "domain"]

    def __init__(self, path: str = "") -> None:
        self.path = path

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

        sync_at = self._sync_datetime(since) if since else None
        saved_items: list[KnowledgeUnit] = []
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
                saved_items.append(unit)

        domain_units = self._domain_units(saved_items)
        if "saved_item" in requested:
            result.units.extend(saved_items)
        if "domain" in requested:
            result.units.extend(domain_units)
        if {"saved_item", "domain"}.issubset(requested):
            result.edges.extend(self._domain_edges(domain_units, saved_items))
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _unit_from_row(self, row: dict[str, Any], path: Path) -> KnowledgeUnit | None:
        url = self._first(row, "url", "given_url", "resolved_url", "item_url")
        if not url:
            return None

        title = self._first(row, "title", "given_title", "resolved_title", "item_title") or url
        added_text = self._first(row, "time_added", "added_at", "created_at")
        added_at = self._parse_datetime(added_text)
        updated_at = self._parse_datetime(
            self._first(row, "time_updated", "updated_at", "time_read", "time_favorited")
        )
        tags = self._parse_tags(self._first(row, "tags", "tag"))
        status = self._normalize_status(self._first(row, "status", "state"))
        domain = self._domain(url)
        favorite = self._is_truthy(self._first(row, "favorite", "is_favorite", "favorited"))
        archived = self._is_archived(row, status)
        read = self._is_read(row, status)
        excerpt = self._first(row, "excerpt", "resolved_excerpt", "description", "summary")
        now = datetime.now(timezone.utc)

        return KnowledgeUnit(
            source_project="pocket_export_csv",
            source_id=self._source_id(url),
            source_entity_type="saved_item",
            title=title,
            content=self._content(title, url, status, favorite, tags, excerpt),
            content_type=ContentType.ARTIFACT,
            metadata=self._metadata(
                title=title,
                url=url,
                added_text=added_text,
                added_at=added_at,
                status=status,
                favorite=favorite,
                archived=archived,
                read=read,
                tags=tags,
                excerpt=excerpt,
                domain=domain,
                source_file=path.name,
            ),
            tags=tags,
            created_at=added_at or updated_at or now,
            updated_at=updated_at or added_at or now,
        )

    def _iter_paths(self) -> list[Path]:
        paths: list[Path] = []
        for source in re.split(r"[\n,]", self.path):
            source = source.strip()
            if not source:
                continue
            path = Path(source).expanduser()
            if path.is_dir():
                paths.extend(sorted(child for child in path.rglob("*.csv") if child.is_file()))
            elif path.is_file():
                paths.append(path)
        return paths

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            return [
                {str(key).strip(): value for key, value in row.items() if key is not None}
                for row in reader
            ]

    def _metadata(
        self,
        *,
        title: str,
        url: str,
        added_text: str,
        added_at: datetime | None,
        status: str,
        favorite: bool,
        archived: bool,
        read: bool,
        tags: list[str],
        excerpt: str,
        domain: str,
        source_file: str,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "title": title,
            "url": url,
            "source_url": url,
            "external_url": url,
            "status": status,
            "favorite": favorite,
            "archived": archived,
            "read": read,
            "tags": tags,
            "domain": domain,
            "source_file": source_file,
        }
        if added_text:
            metadata["time_added"] = added_text
        if added_at:
            metadata["added_at"] = added_at.isoformat()
        if excerpt:
            metadata["excerpt"] = excerpt
        return metadata

    def _content(
        self,
        title: str,
        url: str,
        status: str,
        favorite: bool,
        tags: list[str],
        excerpt: str,
    ) -> str:
        parts = [title, f"URL: {url}"]
        if status:
            parts.append(f"Status: {status}")
        if favorite:
            parts.append("Favorite: true")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if excerpt:
            parts.append(f"Excerpt: {excerpt}")
        return "\n".join(parts)

    def _source_id(self, url: str) -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return f"pocket_export_csv:{digest}"

    def _domain_units(self, saved_items: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for item in saved_items:
            domain = str(item.metadata.get("domain") or "")
            if domain:
                grouped.setdefault(domain, []).append(item)

        units: list[KnowledgeUnit] = []
        for domain, items in sorted(grouped.items()):
            unique_items = sorted({item.source_id: item for item in items}.values(), key=lambda item: item.source_id)
            tags = sorted({tag for item in unique_items for tag in item.metadata.get("tags", [])})
            statuses = sorted({str(item.metadata.get("status")) for item in unique_items if item.metadata.get("status")})
            units.append(
                KnowledgeUnit(
                    source_project="pocket_export_csv",
                    source_id=self._domain_source_id(domain),
                    source_entity_type="domain",
                    title=domain,
                    content=f"Pocket domain: {domain}\nSaved items: {len(unique_items)}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "domain": domain,
                        "item_count": len(unique_items),
                        "saved_item_source_ids": [item.source_id for item in unique_items],
                        "statuses": statuses,
                        "tags": tags,
                        "source_files": sorted({str(item.metadata.get("source_file")) for item in unique_items if item.metadata.get("source_file")}),
                    },
                    tags=["pocket", "domain", domain],
                    created_at=min(item.created_at for item in unique_items),
                    updated_at=max(item.updated_at for item in unique_items),
                )
            )
        return units

    def _domain_edges(self, domains: list[KnowledgeUnit], saved_items: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        domain_ids = {str(domain.metadata.get("domain")): domain.source_id for domain in domains}
        edges: list[KnowledgeEdge] = []
        for item in saved_items:
            domain = str(item.metadata.get("domain") or "")
            domain_id = domain_ids.get(domain)
            if not domain_id:
                continue
            digest = hashlib.sha256(f"{domain_id}|{item.source_id}|contains".encode("utf-8")).hexdigest()[:24]
            edges.append(
                KnowledgeEdge(
                    id=f"pocket-export-csv-domain-contains-{digest}",
                    from_unit_id=domain_id,
                    to_unit_id=item.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": "pocket_export_csv",
                        "from_entity_type": "domain",
                        "to_entity_type": "saved_item",
                        "domain": domain,
                    },
                    created_at=item.created_at,
                )
            )
        return edges

    def _domain_source_id(self, domain: str) -> str:
        digest = hashlib.sha256(domain.encode("utf-8")).hexdigest()[:24]
        return f"pocket_export_csv:domain:{digest}"

    def _domain(self, url: str) -> str:
        parsed = urlparse(url)
        if not parsed.hostname and "://" not in url:
            parsed = urlparse(f"https://{url}")
        host = (parsed.hostname or "").rstrip(".").casefold()
        if not host or any(char.isspace() for char in host):
            return ""
        return host.removeprefix("www.")

    def _parse_tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for tag in re.split(r"[,;|]", value):
            normalized = re.sub(r"\s+", " ", tag.strip().removeprefix("#")).strip().lower()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _normalize_status(self, value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {"0", "active", "unread", "saved"}:
            return "active"
        if normalized == "read":
            return "read"
        if normalized in {"1", "archive", "archived"}:
            return "archived"
        if normalized in {"2", "delete", "deleted"}:
            return "deleted"
        return normalized

    def _is_archived(self, row: dict[str, Any], status: str) -> bool:
        archived = self._first(row, "archived", "is_archived")
        if archived:
            return self._is_truthy(archived)
        return status == "archived"

    def _is_read(self, row: dict[str, Any], status: str) -> bool:
        read = self._first(row, "read", "is_read")
        if read:
            return self._is_truthy(read)
        return bool(self._first(row, "time_read")) or status in {"archived", "read"}

    def _is_truthy(self, value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "favorite", "favorited"}

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        if re.fullmatch(r"\d+(?:\.0+)?", value):
            try:
                return datetime.fromtimestamp(int(float(value)), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        parsed = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
