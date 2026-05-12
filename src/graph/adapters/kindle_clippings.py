"""Adapter for Amazon Kindle My Clippings.txt exports."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class KindleClippingsAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "kindle_clippings"

    @property
    def entity_types(self) -> list[str]:
        return ["book", "clipping"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = self._sync_datetime(since) if since else None
        clippings: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                blocks = self._read_blocks(path)
            except (OSError, UnicodeDecodeError):
                continue
            for block in blocks:
                unit = self._unit_from_block(block, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                clippings.append(unit)

        books = self._book_units(clippings) if "book" in allowed_types else []
        if "book" in allowed_types:
            result.units.extend(books)
        if "clipping" in allowed_types:
            result.units.extend(clippings)
        if "book" in allowed_types and "clipping" in allowed_types:
            result.edges.extend(self._contains_edges(books, clippings))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        path = Path(self.path).expanduser() if self.path else None
        if path is None:
            return []
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(child for child in path.rglob("*.txt") if child.is_file())
        return []

    def _read_blocks(self, path: Path) -> list[list[str]]:
        text = path.read_text(encoding="utf-8-sig")
        return [
            [line.rstrip() for line in block.strip().splitlines()]
            for block in re.split(r"\n=+\s*(?:\n|$)", text)
            if block.strip()
        ]

    def _unit_from_block(self, lines: list[str], source_file: str) -> KnowledgeUnit | None:
        if len(lines) < 2:
            return None

        title, author = self._parse_title(lines[0].strip())
        details = lines[1].strip()
        if not title or not details.startswith("-"):
            return None

        clipping_type = self._parse_type(details)
        page = self._parse_position(details, "page")
        location = self._parse_position(details, "location")
        added_text = self._parse_added(details)
        added_at = self._parse_datetime(added_text)
        text = "\n".join(line for line in lines[3:] if line.strip()).strip()
        if not text and clipping_type != "bookmark":
            return None

        now = datetime.now(timezone.utc)
        created_at = added_at or now
        metadata = {
            "book_title": title,
            "author": author,
            "clipping_type": clipping_type,
            "page": page,
            "location": location,
            "added_at": added_text,
            "source_file": source_file,
        }
        return KnowledgeUnit(
            source_project=SourceProject.KINDLE,
            source_id=f"kindle_clippings:{self._digest(title, author, details, text)}",
            source_entity_type="clipping",
            title=self._title(title, clipping_type, page, location),
            content=text or self._title(title, clipping_type, page, location),
            content_type=ContentType.INSIGHT if clipping_type in {"highlight", "note"} else ContentType.METADATA,
            metadata=metadata,
            tags=[author] if author else [],
            created_at=created_at,
            updated_at=created_at,
        )

    def _parse_title(self, value: str) -> tuple[str, str]:
        match = re.match(r"^(?P<title>.+?)\s+\((?P<author>[^()]*)\)\s*$", value)
        if match:
            return match.group("title").strip(), match.group("author").strip()
        return value.strip(), ""

    def _parse_type(self, details: str) -> str:
        match = re.search(r"\bYour\s+([A-Za-z]+)", details)
        return match.group(1).strip().lower() if match else "clipping"

    def _parse_position(self, details: str, name: str) -> str:
        match = re.search(rf"\b{name}\s+([^|]+)", details, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _parse_added(self, details: str) -> str:
        marker = "Added on "
        if marker not in details:
            return ""
        return details.split(marker, 1)[1].strip()

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        for fmt in ("%A, %B %d, %Y %I:%M:%S %p", "%A, %B %d, %Y %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _title(self, book_title: str, clipping_type: str, page: str, location: str) -> str:
        position = " - ".join(part for part in (f"Page {page}" if page else "", f"Location {location}" if location else "") if part)
        label = clipping_type.title() if clipping_type else "Clipping"
        return f"{book_title}: {label} ({position})" if position else f"{book_title}: {label}"

    def _book_units(self, clippings: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[tuple[str, str], list[KnowledgeUnit]] = {}
        for clipping in clippings:
            key = (str(clipping.metadata.get("book_title", "")), str(clipping.metadata.get("author", "")))
            if key[0]:
                grouped.setdefault(key, []).append(clipping)

        units: list[KnowledgeUnit] = []
        for (title, author), book_clippings in grouped.items():
            source_files = sorted({str(unit.metadata.get("source_file", "")) for unit in book_clippings if unit.metadata.get("source_file")})
            created_at = min(unit.created_at for unit in book_clippings)
            updated_at = max(unit.updated_at for unit in book_clippings)
            locations = [
                location
                for clipping in book_clippings
                if (location := self._location_start(clipping.metadata.get("location"))) is not None
            ]
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.KINDLE,
                    source_id=self._book_source_id(title, author),
                    source_entity_type="book",
                    title=title if not author else f"{title} by {author}",
                    content=title if not author else f"{title}\nAuthor: {author}",
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "book_title": title,
                        "author": author,
                        "clipping_count": len(book_clippings),
                        "highlight_count": sum(1 for unit in book_clippings if unit.metadata.get("clipping_type") == "highlight"),
                        "note_count": sum(1 for unit in book_clippings if unit.metadata.get("clipping_type") == "note"),
                        "location_start": min(locations) if locations else None,
                        "location_end": max(locations) if locations else None,
                        "source_files": source_files,
                        "source_file": source_files,
                    },
                    tags=[author] if author else [],
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return units

    def _contains_edges(self, books: list[KnowledgeUnit], clippings: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        book_ids = {
            (str(book.metadata.get("book_title", "")), str(book.metadata.get("author", ""))): book.source_id
            for book in books
        }
        edges: list[KnowledgeEdge] = []
        seen: set[str] = set()
        for clipping in clippings:
            key = (str(clipping.metadata.get("book_title", "")), str(clipping.metadata.get("author", "")))
            book_source_id = book_ids.get(key)
            if not book_source_id:
                continue
            edge_id = self._edge_id(book_source_id, clipping.source_id)
            if edge_id in seen:
                continue
            seen.add(edge_id)
            edges.append(
                KnowledgeEdge(
                    id=edge_id,
                    from_unit_id=book_source_id,
                    to_unit_id=clipping.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.KINDLE.value,
                        "relation_type": "book_contains_clipping",
                    },
                )
            )
        return edges

    def _book_source_id(self, title: str, author: str) -> str:
        return f"kindle_clippings:book:{self._digest(title, author, '', '')}"

    def _edge_id(self, book_source_id: str, clipping_source_id: str) -> str:
        digest = self._digest(book_source_id, clipping_source_id, "contains", "")
        return f"kindle-clippings-contains-{digest}"

    def _location_start(self, value: object) -> int | None:
        match = re.search(r"\d+", str(value or ""))
        return int(match.group(0)) if match else None

    def _digest(self, title: str, author: str, details: str, text: str) -> str:
        payload = "\n".join((title, author, details, text))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
