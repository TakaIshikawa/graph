"""Adapter for Mailchimp campaigns CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class MailchimpCampaignsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "mailchimp_campaigns_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["campaign"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "campaign" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit is None or (sync_at and ("send_time" not in unit.metadata or unit.updated_at <= sync_at)):
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        campaign_id = first(row, "Campaign ID", "Campaign Id", "ID")
        title = first(row, "Title", "Campaign Title", "Name")
        subject = first(row, "Subject", "Email Subject")
        status = first(row, "Status")
        send_time_text = first(row, "Send Time", "Sent At", "Send Date")
        send_time = parse_datetime(send_time_text)
        audience = first(row, "List", "Audience", "Audience Name", "List Name")
        emails_sent = parse_int(first(row, "Emails Sent", "Recipients", "Sent"))
        opens = parse_int(first(row, "Opens", "Total Opens"))
        clicks = parse_int(first(row, "Clicks", "Total Clicks"))
        unsubscribes = parse_int(first(row, "Unsubscribes", "Unsubscriptions"))
        archive_url = first(row, "Archive URL", "Archive Url", "URL", "Campaign URL")
        if not any([campaign_id, title, subject, status, send_time_text, audience, emails_sent is not None, opens is not None, clicks is not None, unsubscribes is not None, archive_url]):
            return None
        now = datetime.now(timezone.utc)
        timestamp = send_time or now
        metadata = clean_metadata(
            {
                "campaign_id": campaign_id,
                "title": title,
                "subject": subject,
                "status": status,
                "send_time": send_time.isoformat() if send_time else send_time_text,
                "audience": audience,
                "emails_sent": emails_sent,
                "opens": opens,
                "clicks": clicks,
                "unsubscribes": unsubscribes,
                "archive_url": archive_url,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="mailchimp_campaigns_csv",
            source_id=f"mailchimp_campaigns_csv:{campaign_id}" if campaign_id else digest_source_id("mailchimp_campaigns_csv", title, subject, send_time_text, audience, index),
            source_entity_type="campaign",
            title=title or subject or "Mailchimp campaign",
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["mailchimp", "campaign", status, audience] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _content(self, metadata: dict[str, Any]) -> str:
        labels = [("Subject", "subject"), ("Status", "status"), ("Audience", "audience"), ("Emails Sent", "emails_sent"), ("Opens", "opens"), ("Clicks", "clicks"), ("Archive URL", "archive_url")]
        return "\n".join(f"{label}: {metadata[key]}" for label, key in labels if key in metadata)
