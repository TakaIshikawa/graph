"""Adapter for Kindle Vocabulary Builder CSV exports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class KindleVocabularyCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "kindle_vocabulary_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["vocabulary_term"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "vocabulary_term" not in set(entity_types or self.entity_types):
            return result
        sync_at = since.last_sync_at.astimezone(timezone.utc) if since else None
        for path in iter_paths(self.path, {".csv"}):
            for row in read_csv_rows(path):
                unit = self._unit(row, path)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], path: Path) -> KnowledgeUnit | None:
        word = first(row, "Word", "Term", "Vocabulary Word")
        book_title = first(row, "Book Title", "Title")
        if not word:
            return None
        created = parse_datetime(first(row, "Date Added", "Added", "Created")) or datetime.now(timezone.utc)
        lookup = first(row, "Lookup", "Definition", "Meaning")
        context = first(row, "Usage", "Context", "Sentence")
        metadata = clean_metadata(
            {
                "word": word,
                "book_title": book_title,
                "book_author": first(row, "Book Author", "Author", "Authors"),
                "context": context,
                "lookup": lookup,
                "date_added": created.isoformat(),
                "mastery": first(row, "Mastery", "Status", "Learning Status"),
                "source_file": str(path),
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project=self.name,
            source_id=digest_source_id(self.name, word.casefold(), book_title.casefold(), context),
            source_entity_type="vocabulary_term",
            title=f"{word} - {book_title}" if book_title else word,
            content="\n".join(part for part in (f"Word: {word}", f"Book: {book_title}" if book_title else "", f"Context: {context}" if context else "", f"Lookup: {lookup}" if lookup else "") if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["kindle", "vocabulary"],
            created_at=created,
            updated_at=created,
        )
