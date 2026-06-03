"""Adapter for Stack Overflow favorites CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class StackOverflowFavoritesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "stackoverflow_favorites_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["question"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "question" not in entity_types:
            return result
        sync_at = since.last_sync_at.astimezone(timezone.utc) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Question Title", "Title", "Question")
        url = first(row, "URL", "Url", "Link", "Question URL")
        question_id = first(row, "Question ID", "Id", "ID")
        tags = _tags(first(row, "Tags", "Tag List"))
        score = parse_int(first(row, "Score", "Votes"))
        answer_count = parse_int(first(row, "Answer Count", "Answers"))
        accepted = _bool(first(row, "Accepted", "Has Accepted Answer", "Accepted Answer"))
        saved_at = parse_datetime(first(row, "Saved At", "Favorited At", "Bookmarked At"))
        created_at = parse_datetime(first(row, "Created At", "Creation Date"))
        if not any([title, url, question_id]):
            return None
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "question_id": question_id,
                "title": title,
                "url": url,
                "tags": tags,
                "score": score,
                "answer_count": answer_count,
                "accepted": accepted,
                "saved_at": saved_at.isoformat() if saved_at else "",
                "created_at": created_at.isoformat() if created_at else "",
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="stackoverflow_favorites_csv",
            source_id=digest_source_id("stackoverflow_favorites_csv", url or question_id or title or index),
            source_entity_type="question",
            title=title or url or f"Stack Overflow question {question_id}",
            content="\n".join(part for part in [title, f"URL: {url}" if url else "", f"Tags: {', '.join(tags)}" if tags else ""] if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(["stackoverflow", "favorite", *tags])),
            created_at=created_at or saved_at or now,
            updated_at=saved_at or created_at or now,
        )


def _tags(value: str) -> list[str]:
    text = value.strip().strip("[]")
    if not text:
        return []
    parts = re.findall(r"<([^>]+)>", text) or re.split(r"[,;|]", text)
    return [part.strip().strip("'\" ").casefold() for part in parts if part.strip().strip("'\" ")]


def _bool(value: str) -> bool | None:
    text = value.strip().casefold()
    if text in {"true", "yes", "y", "1", "accepted"}:
        return True
    if text in {"false", "no", "n", "0", "none"}:
        return False
    return None
