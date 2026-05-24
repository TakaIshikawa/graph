"""Adapter for Anki notes/cards exported as CSV or TSV."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, parse_int
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class AnkiCardsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "anki_cards_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["card"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "card" not in entity_types:
            return result
        sync_at = since.last_sync_at.astimezone(timezone.utc) if since else None
        for path in iter_paths(self.path, {".csv", ".tsv", ".txt"}):
            for index, row in enumerate(self._read_rows(path)):
                unit = self._unit_from_row(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return []
        sample = text[:2048]
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} or sample.count("\t") > sample.count(",") else ","
        try:
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in csv.DictReader(text.splitlines(), delimiter=delimiter)]
        except csv.Error:
            return []

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        front = first(row, "Front", "Question", "Text")
        back = first(row, "Back", "Answer", "Extra")
        if not front and not back:
            return None
        deck = first(row, "Deck", "Deck Name")
        note_type = first(row, "Note Type", "Model", "Type")
        tags = self._tags(first(row, "Tags", "Tag"))
        due = parse_datetime(first(row, "Due", "Due Date"))
        created = parse_datetime(first(row, "Created", "Created Date"))
        updated = parse_datetime(first(row, "Modified", "Updated"))
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "deck": deck,
                "note_type": note_type,
                "front": front,
                "back": back,
                "tags": tags,
                "due": due.isoformat() if due else "",
                "interval": parse_int(first(row, "Interval", "Interval Days")),
                "ease": parse_int(first(row, "Ease", "Ease Factor")),
                "lapses": parse_int(first(row, "Lapses", "Lapse Count")),
                "source_file": source_file,
                "row": dict(row),
            }
        )
        unit_tags = tags + ([deck] if deck else [])
        return KnowledgeUnit(
            source_project="anki_cards_csv",
            source_id=digest_source_id("anki_cards_csv", first(row, "ID", "Card ID", "Note ID") or front, back, deck, index),
            source_entity_type="card",
            title=front[:120] or "Untitled Anki card",
            content=self._content(front, back, deck, note_type, tags),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=unit_tags,
            created_at=created or due or now,
            updated_at=updated or due or created or now,
        )

    def _tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for tag in value.replace(",", " ").split():
            cleaned = tag.strip()
            if cleaned and cleaned not in tags:
                tags.append(cleaned)
        return tags

    def _content(self, front: str, back: str, deck: str, note_type: str, tags: list[str]) -> str:
        parts = []
        if front:
            parts.append(f"Front: {front}")
        if back:
            parts.append(f"Back: {back}")
        if deck:
            parts.append(f"Deck: {deck}")
        if note_type:
            parts.append(f"Note type: {note_type}")
        if tags:
            parts.append(f"Tags: {' '.join(tags)}")
        return "\n".join(parts)
