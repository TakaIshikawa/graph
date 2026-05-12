"""Adapter for StoryGraph reading history CSV exports."""

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


class StoryGraphReadingHistoryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "storygraph_reading_history_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["book", "read", "author"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or self.entity_types)
        if not allowed.intersection(self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        read_units: list[KnowledgeUnit] = []
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
                read_units.append(unit)

        author_units = self._author_units(read_units)
        if "read" in allowed:
            result.units.extend(read_units)
        if "author" in allowed:
            result.units.extend(author_units)
        if {"author", "read"}.issubset(allowed):
            result.edges.extend(self._author_edges(author_units, read_units))

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if root.is_dir():
            return sorted(child for child in root.rglob("*.csv") if child.is_file())
        return []

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader]

    def _unit_from_row(self, row: dict[str, Any], path: Path) -> KnowledgeUnit | None:
        title = self._first(row, "Title", "title", "Book Title", "book_title")
        if not title:
            return None

        authors = self._split_people(self._first(row, "Authors", "Author", "authors", "author"))
        isbn = self._clean_isbn(self._first(row, "ISBN", "isbn"))
        isbn13 = self._clean_isbn(self._first(row, "ISBN13", "ISBN/UID", "isbn13", "isbn_13"))
        started_text = self._first(row, "Date Started", "Started", "Started At", "started_at", "start_date")
        finished_text = self._first(row, "Date Finished", "Finished", "Finished At", "finished_at", "finish_date")
        date_read_text = self._first(row, "Date Read", "Last Date Read", "Read Date", "date_read", "date read")
        updated_text = self._first(row, "Updated At", "Last Updated", "Date Added", "updated_at", "date_added")
        started_at = self._parse_datetime(started_text)
        finished_at = self._parse_datetime(finished_text) or self._parse_datetime(date_read_text)
        date_read = self._parse_datetime(date_read_text)
        updated_at = self._parse_datetime(updated_text) or date_read
        created_at = date_read or updated_at or datetime.now(timezone.utc)
        rating = self._parse_float(self._first(row, "Star Rating", "Rating", "My Rating", "star_rating", "rating"))
        pages = self._parse_int(self._first(row, "Pages", "Number of Pages", "pages"))
        minutes = self._parse_duration_minutes(
            self._first(
                row,
                "Audiobook Duration",
                "Audio Duration",
                "Duration",
                "Duration Minutes",
                "Minutes",
                "audiobook_duration",
            )
        )
        read_count = self._parse_int(self._first(row, "Read Count", "read_count", "Times Read"))
        moods = self._split_list(self._first(row, "Moods", "moods"))
        shelves = self._split_list(self._first(row, "Tags", "Shelves", "Bookshelves", "tags", "shelves"))
        review = self._first(row, "Review", "My Review", "review", "review_text")
        pace_metadata = self._pace_metadata(started_at, finished_at, pages, minutes)

        metadata = {
            "title": title,
            "authors": authors,
            "isbn": isbn,
            "isbn13": isbn13,
            "started_at": started_at.isoformat() if started_at else "",
            "finished_at": finished_at.isoformat() if finished_at else "",
            "date_read": date_read.isoformat() if date_read else "",
            "read_count": read_count,
            "rating": rating,
            "moods": moods,
            "tags": shelves,
            "shelves": shelves,
            "pages": pages,
            "duration_minutes": minutes,
            **pace_metadata,
            "review": review,
            "source_file": str(path),
            "row": dict(row),
        }
        return KnowledgeUnit(
            source_project=SourceProject.STORYGRAPH_READING_HISTORY_CSV,
            source_id=self._source_id(title, authors, isbn13 or isbn, date_read_text, read_count, row),
            source_entity_type="read",
            title=self._format_title(title, authors),
            content=self._content(title, authors, date_read, rating, shelves, moods, review),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=self._dedupe(["storygraph", *shelves, *moods]),
            created_at=created_at,
            updated_at=updated_at or created_at,
        )

    def _source_id(
        self,
        title: str,
        authors: list[str],
        isbn: str,
        date_read: str,
        read_count: int | None,
        row: dict[str, Any],
    ) -> str:
        explicit = self._first(row, "ID", "Book ID", "Reading ID", "id", "book_id")
        if explicit:
            raw = explicit
        else:
            raw = "|".join([isbn, title, ";".join(authors), date_read, "" if read_count is None else str(read_count)])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"storygraph_reading_history_csv:{digest}"

    def _content(
        self,
        title: str,
        authors: list[str],
        date_read: datetime | None,
        rating: float | None,
        shelves: list[str],
        moods: list[str],
        review: str,
    ) -> str:
        parts = [f"Title: {title}"]
        if authors:
            parts.append(f"Authors: {', '.join(authors)}")
        if date_read:
            parts.append(f"Date read: {date_read.date().isoformat()}")
        if rating is not None:
            parts.append(f"Rating: {rating:g}/5")
        if shelves:
            parts.append(f"Tags: {', '.join(shelves)}")
        if moods:
            parts.append(f"Moods: {', '.join(moods)}")
        if review:
            parts.append(f"\nReview:\n{review}")
        return "\n".join(parts)

    def _format_title(self, title: str, authors: list[str]) -> str:
        return f"{title} by {', '.join(authors)}" if authors else title

    def _author_units(self, reads: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for read in reads:
            for author in read.metadata.get("authors") or []:
                key = self._normalized_author(author)
                if not key:
                    continue
                grouped.setdefault(key, []).append(read)
                names.setdefault(key, str(author).strip())

        units: list[KnowledgeUnit] = []
        for key, author_reads in sorted(grouped.items()):
            unique_reads = sorted({read.source_id: read for read in author_reads}.values(), key=lambda read: read.source_id)
            titles = sorted({str(read.metadata.get("title") or read.title) for read in unique_reads})
            read_source_ids = [read.source_id for read in unique_reads]
            source_files = sorted({str(read.metadata.get("source_file")) for read in unique_reads if read.metadata.get("source_file")})
            author = names[key]
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.STORYGRAPH_READING_HISTORY_CSV,
                    source_id=self._author_source_id(key),
                    source_entity_type="author",
                    title=author,
                    content="\n".join([f"Author: {author}", f"Books read: {len(unique_reads)}"]),
                    content_type=ContentType.METADATA,
                    metadata={
                        "author": author,
                        "normalized_author": key,
                        "book_count": len(unique_reads),
                        "read_source_ids": read_source_ids,
                        "titles": titles,
                        "source_files": source_files,
                    },
                    tags=["storygraph", "author"],
                    created_at=min(read.created_at for read in unique_reads),
                    updated_at=max(read.updated_at for read in unique_reads),
                )
            )
        return units

    def _author_edges(self, authors: list[KnowledgeUnit], reads: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        author_ids = {str(author.metadata.get("normalized_author")): author.source_id for author in authors}
        edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str]] = set()
        for read in reads:
            for author in read.metadata.get("authors") or []:
                author_id = author_ids.get(self._normalized_author(str(author)))
                if not author_id or (author_id, read.source_id) in seen:
                    continue
                seen.add((author_id, read.source_id))
                edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(author_id, read.source_id),
                        from_unit_id=author_id,
                        to_unit_id=read.source_id,
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.STORYGRAPH_READING_HISTORY_CSV.value,
                            "from_entity_type": "author",
                            "to_entity_type": "read",
                            "author": author,
                        },
                        created_at=read.created_at,
                    )
                )
        return edges

    def _normalized_author(self, author: str) -> str:
        return " ".join(author.casefold().split())

    def _author_source_id(self, normalized_author: str) -> str:
        digest = hashlib.sha256(normalized_author.encode("utf-8")).hexdigest()[:24]
        return f"storygraph_reading_history_csv:author:{digest}"

    def _edge_id(self, author_id: str, read_id: str) -> str:
        digest = hashlib.sha256(f"{author_id}|{read_id}|contains".encode("utf-8")).hexdigest()[:24]
        return f"storygraph-reading-history-author-contains-{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).lower(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.lower())
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _split_people(self, value: str) -> list[str]:
        return self._dedupe(part.strip() for part in re.split(r"\s*(?:,|;|\band\b)\s*", value) if part.strip())

    def _split_list(self, value: str) -> list[str]:
        return self._dedupe(part.strip().lower() for part in re.split(r"[,;|]", value) if part.strip())

    def _dedupe(self, values: Any) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in result:
                result.append(text)
        return result

    def _clean_isbn(self, value: str) -> str:
        return value.strip().strip('="').replace("-", "").replace(" ", "")

    def _parse_int(self, value: str) -> int | None:
        if not value:
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    def _parse_float(self, value: str) -> float | None:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _parse_duration_minutes(self, value: str) -> int | None:
        if not value:
            return None
        text = value.strip().lower().replace(",", "")
        if re.fullmatch(r"\d+(?::\d{1,2}){1,2}", text):
            parts = [int(part) for part in text.split(":")]
            if len(parts) == 2:
                return parts[0] * 60 + parts[1]
            return parts[0] * 60 + parts[1] + round(parts[2] / 60)
        match = re.search(r"\d+(?:\.\d+)?", text)
        if not match:
            return None
        number = float(match.group(0))
        if "hour" in text or re.search(r"\bhrs?\b", text):
            return int(round(number * 60))
        return int(round(number))

    def _pace_metadata(
        self,
        started_at: datetime | None,
        finished_at: datetime | None,
        pages: int | None,
        minutes: int | None,
    ) -> dict[str, Any]:
        reading_days = None
        if started_at and finished_at:
            reading_days = max(1, (finished_at.date() - started_at.date()).days + 1)
        metadata: dict[str, Any] = {"completion_bucket": self._completion_bucket(reading_days)}
        if reading_days is not None:
            metadata["reading_days"] = reading_days
            if pages is not None and pages > 0:
                metadata["pages_per_day"] = round(pages / reading_days, 2)
            if minutes is not None and minutes > 0:
                metadata["minutes_per_day"] = round(minutes / reading_days, 2)
        return metadata

    def _completion_bucket(self, reading_days: int | None) -> str:
        if reading_days is None:
            return "unknown"
        if reading_days <= 1:
            return "same_day"
        if reading_days <= 7:
            return "week"
        if reading_days <= 31:
            return "month"
        return "long_read"

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        text = value.strip()
        for candidate in (text, f"{text}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                pass
        for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
