"""Adapter for Trello card CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class TrelloCardsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "trello_cards_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["card"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "card" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        card_id = first(row, "card id", "id", "short id")
        name = first(row, "card name", "name", "title")
        description = first(row, "description", "desc")
        url = first(row, "url", "card url")
        if not any([card_id, name, description, url]):
            return None
        board = first(row, "board", "board name")
        list_name = first(row, "list", "list name")
        labels = split_values(first(row, "labels", "label names"))
        members = split_values(first(row, "members", "member names"))
        due = parse_datetime(first(row, "due", "due date"))
        updated = parse_datetime(first(row, "date last activity", "last activity", "updated_at")) or due or datetime.now(timezone.utc)
        closed = _bool(first(row, "closed", "archived"))
        metadata = clean_metadata({"card_id": card_id, "name": name, "description": description, "url": url, "board": board, "list": list_name, "labels": labels, "members": members, "due": due.isoformat() if due else first(row, "due", "due date"), "date_last_activity": updated.isoformat(), "closed": closed, "source_file": source_file})
        tags = [tag for tag in dict.fromkeys(["trello", "card", board, list_name, *labels]) if tag]
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{card_id}" if card_id else digest_source_id(self.name, name, url, index), source_entity_type="card", title=name or f"Trello card {card_id or index + 1}", content=_lines(name, description, board, list_name, url), content_type=ContentType.ARTIFACT, metadata=metadata, tags=tags, created_at=updated, updated_at=updated)


def _bool(value: str) -> bool | None:
    text = value.strip().casefold()
    if text in {"true", "yes", "1", "closed", "archived"}:
        return True
    if text in {"false", "no", "0", "open"}:
        return False
    return None


def _lines(*parts: str) -> str:
    labels = ("", "", "Board: ", "List: ", "URL: ")
    return "\n".join(f"{prefix}{part}" for prefix, part in zip(labels, parts, strict=False) if part)
