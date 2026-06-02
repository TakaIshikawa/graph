"""Adapter for Google Forms response CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GoogleFormsResponsesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_forms_responses_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["form_response"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "form_response" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
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
        timestamp_text = first(row, "Timestamp", "Submitted At", "Created At")
        timestamp = parse_datetime(timestamp_text)
        email = first(row, "Email Address", "Respondent Email", "Email")
        response_id = first(row, "Response ID", "Response Id", "ID")
        reserved = {"timestamp", "submittedat", "createdat", "emailaddress", "respondentemail", "email", "responseid", "id"}
        answers = {key: str(value).strip() for key, value in row.items() if key and key.strip().replace(" ", "").lower() not in reserved and str(value).strip()}
        if not answers:
            return None
        now = datetime.now(timezone.utc)
        title = f"Google Forms response {response_id or index + 1}"
        metadata = clean_metadata(
            {
                "response_id": response_id,
                "timestamp": timestamp.isoformat() if timestamp else timestamp_text,
                "respondent_email": email,
                "answers": answers,
                "source_file": source_file,
            }
        )
        content = "\n".join([title, *[f"{question}: {answer}" for question, answer in answers.items()]])
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_FORMS_RESPONSES_CSV,
            source_id=digest_source_id("google_forms_responses_csv", response_id or timestamp_text or source_file, email, index if not response_id else ""),
            source_entity_type="form_response",
            title=title,
            content=content,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["google_forms", "form_response"],
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )
