"""Adapter for Libby loans and history CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class LibbyLoansCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "libby_loans_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["loan"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types and "loan" not in entity_types:
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
                result.edges.extend(self._edges_for_unit(unit))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
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
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in csv.DictReader(handle)]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = self._first(row, "Title", "Name")
        author = self._first(row, "Author", "Authors")
        library = self._first(row, "Library", "Library Name")
        borrowed = self._parse_date(self._first(row, "Borrowed Date", "Borrowed", "Checkout Date"))
        returned = self._parse_date(self._first(row, "Returned Date", "Returned", "Return Date"))
        isbn = self._first(row, "ISBN", "ISBN13", "ISBN 13")
        if not title and not isbn:
            return None
        subjects = self._split(self._first(row, "Subjects", "Subject", "Categories"))
        fmt = self._first(row, "Format", "Media Format")
        metadata = {
            "title": title,
            "author": author,
            "format": fmt,
            "library": library,
            "borrowed_at": borrowed.isoformat() if borrowed else self._first(row, "Borrowed Date", "Borrowed", "Checkout Date"),
            "returned_at": returned.isoformat() if returned else self._first(row, "Returned Date", "Returned", "Return Date"),
            "series": self._first(row, "Series", "Series Name"),
            "subjects": subjects,
            "isbn": isbn,
            "rating": self._first(row, "Rating", "My Rating"),
            "source_file": source_file,
            "row": dict(row),
        }
        now = datetime.now(timezone.utc)
        tags = ["libby", "loan", *subjects]
        if fmt:
            tags.append(fmt.lower())
        return KnowledgeUnit(
            source_project=SourceProject.LIBBY_LOANS_CSV,
            source_id=self._source_id(title, author, library, borrowed, isbn),
            source_entity_type="loan",
            title=title or isbn,
            content=self._content(title, author, metadata),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(item for item in tags if item)),
            created_at=borrowed or now,
            updated_at=returned or borrowed or now,
        )

    def _edges_for_unit(self, unit: KnowledgeUnit) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for kind in ("author", "library", "series"):
            value = unit.metadata.get(kind)
            if value:
                edges.append(self._edge(unit.source_id, f"libby:{kind}:{value}", kind, str(value)))
        return edges

    def _edge(self, source_id: str, target: str, kind: str, value: str) -> KnowledgeEdge:
        digest = hashlib.sha256(f"{source_id}|{target}".encode("utf-8")).hexdigest()[:24]
        return KnowledgeEdge(id=f"libby_loans_csv:{digest}", from_unit_id=source_id, to_unit_id=target, relation=EdgeRelation.RELATES_TO, source=EdgeSource.SOURCE, metadata={"kind": kind, "value": value})

    def _content(self, title: str, author: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        if author:
            parts.append(f"Author: {author}")
        for key, label in (("format", "Format"), ("library", "Library"), ("borrowed_at", "Borrowed"), ("returned_at", "Returned"), ("series", "Series"), ("isbn", "ISBN"), ("rating", "Rating")):
            if metadata.get(key):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(item for item in parts if item)

    def _source_id(self, title: str, author: str, library: str, borrowed: datetime | None, isbn: str) -> str:
        raw = isbn or "|".join([title, author, library, borrowed.isoformat() if borrowed else ""])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"libby_loans_csv:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _split(self, value: str) -> list[str]:
        items: list[str] = []
        for item in re.split(r"[,;|]", value or ""):
            text = item.strip()
            if text and text not in items:
                items.append(text)
        return items

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _parse_date(self, value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        for candidate in (text, text.replace("Z", "+00:00")):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate))
            except ValueError:
                pass
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
