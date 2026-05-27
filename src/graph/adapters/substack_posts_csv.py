"""Adapter for Substack posts CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SubstackPostsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "substack_posts_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["post"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "post" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "title", "Title")
        subtitle = first(row, "subtitle", "Subtitle")
        body = first(row, "body", "Body", "description", "Description")
        if not any((title, subtitle, body)):
            return None
        url = first(row, "canonical_url", "Canonical URL", "url", "URL")
        date_text = first(row, "post_date", "Post Date", "published_at", "Published At")
        posted_at = parse_datetime(date_text)
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "subtitle": subtitle,
                "canonical_url": url,
                "post_date": posted_at.isoformat() if posted_at else date_text,
                "audience": first(row, "audience", "Audience"),
                "type": first(row, "type", "Type"),
                "likes": parse_int(first(row, "likes", "Likes")),
                "comments": parse_int(first(row, "comments", "Comments")),
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="substack_posts_csv",
            source_id=digest_source_id("substack_posts_csv", url or title, "" if url else date_text or index),
            source_entity_type="post",
            title=title or subtitle or "Substack post",
            content="\n".join(part for part in [title, subtitle, body, f"URL: {url}" if url else ""] if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            created_at=posted_at or now,
            updated_at=posted_at or now,
        )
