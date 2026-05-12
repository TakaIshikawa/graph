"""Adapter for Amazon Kindle My Clippings.txt exports."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
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
        if "clipping" in allowed_types:
            result.edges.extend(self._note_highlight_edges(clippings))
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
            return sorted(
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in {".txt", ".html", ".htm"}
            )
        return []

    def _read_blocks(self, path: Path) -> list[list[str]]:
        if path.suffix.lower() in {".html", ".htm"}:
            return self._read_html_blocks(path)
        text = path.read_text(encoding="utf-8-sig")
        return [
            [line.rstrip() for line in block.strip().splitlines()]
            for block in re.split(r"\n=+\s*(?:\n|$)", text)
            if block.strip()
        ]

    def _read_html_blocks(self, path: Path) -> list[list[str]]:
        parser = _KindleNotebookHtmlParser()
        parser.feed(path.read_text(encoding="utf-8-sig"))
        parser.close()
        parser.flush()
        text = "\n".join(line for line in parser.lines if line)
        title, author = self._parse_html_book_title(text)
        blocks: list[list[str]] = []
        for match in re.finditer(
            r"(?P<type>Highlight|Note|Bookmark)\s*(?:\((?P<position>[^)]*)\))?\s*(?P<body>.*?)(?=\n(?:Highlight|Note|Bookmark)\b|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            clipping_type = match.group("type").lower()
            position = (match.group("position") or "").strip()
            body_lines = [line.strip() for line in match.group("body").splitlines() if line.strip()]
            added_at = ""
            content_lines: list[str] = []
            for line in body_lines:
                parsed_added = self._parse_html_added(line)
                if parsed_added:
                    added_at = parsed_added
                    continue
                if not self._is_html_book_heading(line, title):
                    content_lines.append(line)
            details = f"- Your {clipping_type.title()} {self._html_position_details(position)}"
            if added_at:
                details = f"{details} | Added on {added_at}"
            blocks.append([self._html_title_line(title, author), details, "", *content_lines])
        return blocks

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
        for fmt in (
            "%A, %B %d, %Y %I:%M:%S %p",
            "%A, %B %d, %Y %H:%M:%S",
            "%B %d, %Y %I:%M:%S %p",
            "%B %d, %Y %H:%M:%S",
            "%b %d, %Y %I:%M:%S %p",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
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
            key = self._book_identity(clipping.metadata)
            if key[0]:
                grouped.setdefault(key, []).append(clipping)

        units: list[KnowledgeUnit] = []
        for (title, author), book_clippings in grouped.items():
            first = book_clippings[0]
            display_title = str(first.metadata.get("book_title") or title)
            display_author = str(first.metadata.get("author") or author)
            source_files = sorted({str(unit.metadata.get("source_file", "")) for unit in book_clippings if unit.metadata.get("source_file")})
            created_at = min(unit.created_at for unit in book_clippings)
            updated_at = max(unit.updated_at for unit in book_clippings)
            locations = [
                location
                for clipping in book_clippings
                if (location := self._location_start(clipping.metadata.get("location"))) is not None
            ]
            pages = [
                page
                for clipping in book_clippings
                if (page := self._location_start(clipping.metadata.get("page"))) is not None
            ]
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.KINDLE,
                    source_id=self._book_source_id(display_title, display_author),
                    source_entity_type="book",
                    title=display_title if not display_author else f"{display_title} by {display_author}",
                    content=display_title if not display_author else f"{display_title}\nAuthor: {display_author}",
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "book_title": display_title,
                        "author": display_author,
                        "clipping_count": len(book_clippings),
                        "highlight_count": sum(1 for unit in book_clippings if unit.metadata.get("clipping_type") == "highlight"),
                        "note_count": sum(1 for unit in book_clippings if unit.metadata.get("clipping_type") == "note"),
                        "bookmark_count": sum(1 for unit in book_clippings if unit.metadata.get("clipping_type") == "bookmark"),
                        "first_clipped_at": created_at.isoformat(),
                        "last_clipped_at": updated_at.isoformat(),
                        "page_start": min(pages) if pages else None,
                        "page_end": max(pages) if pages else None,
                        "location_start": min(locations) if locations else None,
                        "location_end": max(locations) if locations else None,
                        "source_files": source_files,
                        "source_file": source_files,
                    },
                    tags=[display_author] if display_author else [],
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return units

    def _contains_edges(self, books: list[KnowledgeUnit], clippings: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        book_ids = {
            self._book_identity(book.metadata): book.source_id
            for book in books
        }
        edges: list[KnowledgeEdge] = []
        seen: set[str] = set()
        for clipping in clippings:
            key = self._book_identity(clipping.metadata)
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

    def _note_highlight_edges(self, clippings: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        highlights = [unit for unit in clippings if unit.metadata.get("clipping_type") == "highlight"]
        notes = [unit for unit in clippings if unit.metadata.get("clipping_type") == "note"]
        edges: list[KnowledgeEdge] = []
        seen: set[str] = set()
        for note in notes:
            match = self._matching_highlight(note, highlights)
            if match is None:
                continue
            highlight, strategy = match
            edge_id = self._annotation_edge_id(note.source_id, highlight.source_id, strategy)
            if edge_id in seen:
                continue
            seen.add(edge_id)
            edges.append(
                KnowledgeEdge(
                    id=edge_id,
                    from_unit_id=note.source_id,
                    to_unit_id=highlight.source_id,
                    relation=EdgeRelation.RELATES_TO,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.KINDLE.value,
                        "relation_type": "note_references_highlight",
                        "book_title": note.metadata.get("book_title"),
                        "location": note.metadata.get("location"),
                        "matched_location": highlight.metadata.get("location"),
                        "match_strategy": strategy,
                    },
                )
            )
        return edges

    def _matching_highlight(
        self, note: KnowledgeUnit, highlights: list[KnowledgeUnit]
    ) -> tuple[KnowledgeUnit, str] | None:
        note_key = (str(note.metadata.get("book_title") or ""), str(note.metadata.get("author") or ""))
        note_range = self._location_range(note.metadata.get("location"))
        if note_range is None:
            return None
        candidates: list[tuple[int, str, KnowledgeUnit]] = []
        for highlight in highlights:
            highlight_key = (str(highlight.metadata.get("book_title") or ""), str(highlight.metadata.get("author") or ""))
            if highlight_key != note_key:
                continue
            highlight_range = self._location_range(highlight.metadata.get("location"))
            if highlight_range is None:
                continue
            distance = self._location_distance(note_range, highlight_range)
            if distance == 0:
                strategy = "exact" if note_range == highlight_range else "overlap"
                candidates.append((0, strategy, highlight))
            elif distance <= 5:
                candidates.append((distance, "nearby", highlight))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[2].source_id))
        _, strategy, highlight = candidates[0]
        return highlight, strategy

    def _book_source_id(self, title: str, author: str) -> str:
        title_key, author_key = self._book_identity({"book_title": title, "author": author})
        return f"kindle_clippings:book:{self._digest(title_key, author_key, '', '')}"

    def _book_identity(self, metadata: dict) -> tuple[str, str]:
        title = " ".join(str(metadata.get("book_title", "")).strip().casefold().split())
        author = " ".join(str(metadata.get("author", "")).strip().casefold().split())
        return title, author

    def _edge_id(self, book_source_id: str, clipping_source_id: str) -> str:
        digest = self._digest(book_source_id, clipping_source_id, "contains", "")
        return f"kindle-clippings-contains-{digest}"

    def _annotation_edge_id(self, note_source_id: str, highlight_source_id: str, strategy: str) -> str:
        digest = self._digest(note_source_id, highlight_source_id, strategy, "annotation")
        return f"kindle-clippings-note-highlight-{digest}"

    def _location_start(self, value: object) -> int | None:
        match = re.search(r"\d+", str(value or ""))
        return int(match.group(0)) if match else None

    def _location_range(self, value: object) -> tuple[int, int] | None:
        numbers = [int(match.group(0)) for match in re.finditer(r"\d+", str(value or ""))]
        if not numbers:
            return None
        if len(numbers) == 1:
            return (numbers[0], numbers[0])
        return (min(numbers[0], numbers[1]), max(numbers[0], numbers[1]))

    def _location_distance(self, left: tuple[int, int], right: tuple[int, int]) -> int:
        if left[0] <= right[1] and right[0] <= left[1]:
            return 0
        if left[1] < right[0]:
            return right[0] - left[1]
        return left[0] - right[1]

    def _digest(self, title: str, author: str, details: str, text: str) -> str:
        payload = "\n".join((title, author, details, text))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    def _parse_html_book_title(self, text: str) -> tuple[str, str]:
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned and cleaned.lower() not in {"highlight", "note", "bookmark"}:
                return self._parse_title(cleaned)
        return "", ""

    def _html_title_line(self, title: str, author: str) -> str:
        return f"{title} ({author})" if author else title

    def _html_position_details(self, position: str) -> str:
        parts: list[str] = []
        page = self._html_position_value(position, "page")
        location = self._html_position_value(position, "location")
        if page:
            parts.append(f"on page {page}")
        if location:
            parts.append(f"at location {location}")
        return " | ".join(parts)

    def _html_position_value(self, position: str, name: str) -> str:
        match = re.search(rf"\b{name}\s+([\w-]+)", position, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _parse_html_added(self, line: str) -> str:
        match = re.search(r"(?:Added on|Created|Last annotated)\s*:?\s*(.+)", line, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _is_html_book_heading(self, line: str, title: str) -> bool:
        return bool(title and line.strip().casefold() == title.casefold())


class _KindleNotebookHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self._current: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"br", "p", "div", "section", "article", "h1", "h2", "h3", "li"}:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"p", "div", "section", "article", "h1", "h2", "h3", "li"}:
            self._flush()

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self._current.append(text)

    def _flush(self) -> None:
        if self._current:
            self.lines.append(" ".join(self._current).strip())
            self._current = []

    def flush(self) -> None:
        self._flush()
