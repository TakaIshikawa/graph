"""Adapter for Calendly scheduled event CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class CalendlyEventsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "calendly_events_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["scheduled_event"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "scheduled_event" not in entity_types:
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
        event_id = first(row, "Event UUID", "UUID", "Event ID", "ID")
        event_type = first(row, "Event Type", "Event Type Name")
        invitee = first(row, "Invitee Name", "Name")
        email = first(row, "Invitee Email", "Email")
        start = parse_datetime(first(row, "Start Time", "Start Date Time", "Start"))
        end = parse_datetime(first(row, "End Time", "End Date Time", "End"))
        created = parse_datetime(first(row, "Created At", "Created"))
        if not any([event_id, event_type, invitee, email, start]):
            return None
        reserved = {"eventuuid", "uuid", "eventid", "id", "eventtype", "eventtypename", "inviteename", "name", "inviteeemail", "email", "starttime", "startdatetime", "start", "endtime", "enddatetime", "end", "timezone", "status", "cancellationreason", "location", "joinurl", "createdat", "created"}
        qa = {key: str(value).strip() for key, value in row.items() if key and key.strip().replace(" ", "").lower() not in reserved and str(value).strip()}
        location = first(row, "Location", "Join URL", "Location/Join URL")
        status = first(row, "Status")
        cancel_reason = first(row, "Cancellation Reason", "Cancel Reason")
        timezone_text = first(row, "Timezone", "Time Zone")
        metadata = clean_metadata(
            {
                "event_id": event_id,
                "event_type": event_type,
                "invitee_name": invitee,
                "invitee_email": email,
                "start_time": start.isoformat() if start else "",
                "end_time": end.isoformat() if end else "",
                "timezone": timezone_text,
                "status": status,
                "cancellation_reason": cancel_reason,
                "location": location,
                "created_at": created.isoformat() if created else "",
                "questions": qa,
                "source_file": source_file,
            }
        )
        now = datetime.now(timezone.utc)
        title = " - ".join(part for part in [event_type, invitee] if part) or event_id or "Calendly event"
        return KnowledgeUnit(
            source_project=SourceProject.CALENDLY_EVENTS_CSV,
            source_id=digest_source_id("calendly_events_csv", event_id or title, start.isoformat() if start else "", index if not event_id else ""),
            source_entity_type="scheduled_event",
            title=title,
            content="\n".join(part for part in [title, f"Invitee: {invitee} {email}".strip(), f"Status: {status}" if status else "", f"Location: {location}" if location else "", *[f"{q}: {a}" for q, a in qa.items()]] if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["calendly", "scheduled_event"],
            created_at=created or start or now,
            updated_at=start or created or now,
        )
