"""Adapter for Gmail label CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GmailLabelsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "gmail_labels_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["email_label"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "email_label" not in entity_types:
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
        message_id = first(row, "message id", "message_id", "id")
        thread_id = first(row, "thread id", "thread_id")
        subject = first(row, "subject", "title")
        snippet = first(row, "snippet", "body")
        if not any([message_id, thread_id, subject, snippet]):
            return None
        labels = split_values(first(row, "labels", "label"))
        date = parse_datetime(first(row, "date", "sent at", "created_at")) or datetime.now(timezone.utc)
        metadata = clean_metadata({"message_id": message_id, "thread_id": thread_id, "subject": subject, "from": first(row, "from", "sender"), "to": first(row, "to", "recipient", "recipients"), "date": date.isoformat(), "labels": labels, "snippet": snippet, "url": first(row, "url", "link"), "source_file": source_file})
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{message_id}" if message_id else digest_source_id(self.name, thread_id, subject, date, index), source_entity_type="email_label", title=subject or f"Gmail message {message_id or index + 1}", content=_content(subject, snippet, metadata), content_type=ContentType.ARTIFACT, metadata=metadata, tags=[tag for tag in dict.fromkeys(["gmail", "email_label", *labels]) if tag], created_at=date, updated_at=date)


def _content(subject: str, snippet: str, metadata: dict[str, Any]) -> str:
    return "\n".join(part for part in (subject, snippet, f"From: {metadata.get('from')}" if metadata.get("from") else "", f"Labels: {', '.join(metadata.get('labels', []))}" if metadata.get("labels") else "") if part)
