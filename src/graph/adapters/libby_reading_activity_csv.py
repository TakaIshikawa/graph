"""Adapter for Libby reading activity CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class LibbyReadingActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "libby_reading_activity_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["library_activity"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "library_activity" not in set(entity_types or self.entity_types):
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        units: dict[str, KnowledgeUnit] = {}
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows, start=1):
                unit = self._unit(row, path.name, index)
                if unit is None:
                    continue
                activity_at = self._activity_at(unit)
                if sync_at and activity_at and activity_at <= sync_at:
                    continue
                units.setdefault(unit.source_id, unit)

        result.units = sorted(units.values(), key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, row_index: int) -> KnowledgeUnit | None:
        if not any(str(value).strip() for value in row.values() if value is not None):
            return None

        title = first(row, "Title", "Book Title", "Name")
        authors = split_values(first(row, "Author", "Authors", "Creator"))
        action = self._action(row)
        borrowed_at = parse_datetime(first(row, "Borrowed Date", "Borrowed", "Checkout Date", "Checked Out At"))
        due_at = parse_datetime(first(row, "Due Date", "Due", "Expires", "Expires At"))
        returned_at = parse_datetime(first(row, "Returned Date", "Returned", "Return Date"))
        placed_at = parse_datetime(first(row, "Placed Date", "Placed", "Hold Placed Date"))
        activity_at = self._event_datetime(action, borrowed_at, due_at, returned_at, placed_at) or parse_datetime(first(row, "Activity Date", "Date", "Timestamp"))
        loan_id = first(row, "Loan ID", "Libby Loan ID", "Hold ID", "Libby Hold ID", "ID")
        if not any([title, authors, loan_id, activity_at]):
            return None

        metadata = {
            "title": title,
            "authors": authors,
            "author": "; ".join(authors),
            "format": first(row, "Format", "Media Format", "Type"),
            "library": first(row, "Library", "Library Name"),
            "card": first(row, "Card", "Library Card", "Card Name"),
            "action": action,
            "borrowed_at": borrowed_at.isoformat() if borrowed_at else first(row, "Borrowed Date", "Borrowed", "Checkout Date", "Checked Out At"),
            "due_at": due_at.isoformat() if due_at else first(row, "Due Date", "Due", "Expires", "Expires At"),
            "returned_at": returned_at.isoformat() if returned_at else first(row, "Returned Date", "Returned", "Return Date"),
            "placed_at": placed_at.isoformat() if placed_at else first(row, "Placed Date", "Placed", "Hold Placed Date"),
            "activity_at": activity_at.isoformat() if activity_at else "",
            "loan_id": loan_id,
            "source_file": source_file,
            "row_index": row_index,
            "raw_record": dict(row),
        }
        now = datetime.now(timezone.utc)
        name = self._title(title, action)
        return KnowledgeUnit(
            source_project="libby_reading_activity_csv",
            source_id=self._source_id(row, loan_id, action, activity_at),
            source_entity_type="library_activity",
            title=name,
            content=self._content(title or "Untitled", authors, metadata),
            content_type=ContentType.METADATA,
            metadata=clean_metadata(metadata),
            tags=list(dict.fromkeys(tag for tag in ["libby", "library-activity", action, metadata["format"], metadata["library"]] if tag)),
            created_at=activity_at or borrowed_at or placed_at or returned_at or now,
            updated_at=activity_at or returned_at or borrowed_at or placed_at or now,
        )

    def _action(self, row: dict[str, Any]) -> str:
        explicit = first(row, "Activity", "Action", "Event", "Type", "Status").strip().lower()
        if explicit:
            if "borrow" in explicit or "checkout" in explicit or "loan" in explicit:
                return "borrow"
            if "return" in explicit:
                return "return"
            if "renew" in explicit:
                return "renew"
            if "hold" in explicit or "place" in explicit:
                return "hold"
            return explicit.replace(" ", "_")
        if first(row, "Returned Date", "Returned", "Return Date"):
            return "return"
        if first(row, "Placed Date", "Placed", "Hold Placed Date"):
            return "hold"
        return "borrow"

    def _event_datetime(
        self,
        action: str,
        borrowed_at: datetime | None,
        due_at: datetime | None,
        returned_at: datetime | None,
        placed_at: datetime | None,
    ) -> datetime | None:
        if action == "return":
            return returned_at or borrowed_at or placed_at or due_at
        if action == "hold":
            return placed_at or borrowed_at or due_at or returned_at
        if action == "renew":
            return borrowed_at or due_at or returned_at or placed_at
        return borrowed_at or placed_at or returned_at or due_at

    def _source_id(self, row: dict[str, Any], loan_id: str, action: str, activity_at: datetime | None) -> str:
        if loan_id:
            return digest_source_id("libby_reading_activity_csv", loan_id, action, activity_at.isoformat() if activity_at else "")
        return digest_source_id("libby_reading_activity_csv", dict(sorted((str(key), str(value)) for key, value in row.items())))

    def _activity_at(self, unit: KnowledgeUnit) -> datetime | None:
        return parse_datetime(unit.metadata.get("activity_at"))

    def _title(self, title: str, action: str) -> str:
        label = {"borrow": "Borrowed", "return": "Returned", "renew": "Renewed", "hold": "Held"}.get(action, action.title())
        return f"{label}: {title}" if title else label

    def _content(self, title: str, authors: list[str], metadata: dict[str, Any]) -> str:
        parts = [title]
        if authors:
            parts.append(f"Authors: {', '.join(authors)}")
        for key, label in (
            ("action", "Action"),
            ("format", "Format"),
            ("library", "Library"),
            ("card", "Card"),
            ("borrowed_at", "Borrowed"),
            ("due_at", "Due"),
            ("returned_at", "Returned"),
            ("placed_at", "Placed"),
        ):
            if metadata.get(key) not in ("", None, []):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
