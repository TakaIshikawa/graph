"""Adapter for LinkedIn saved post CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class LinkedInSavedPostsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "linkedin_saved_posts_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["saved_post"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "saved_post" not in entity_types:
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
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        post_url = first(row, "Post URL", "URL", "Link")
        text = first(row, "Text", "Post Text", "Content")
        if not post_url and not text:
            return None
        saved_at = parse_datetime(first(row, "Saved Date", "Saved At", "Date")) or datetime.now(timezone.utc)
        author = first(row, "Author", "Name")
        author_profile_url = first(row, "Author Profile URL", "Profile URL")
        reaction_count = parse_int(first(row, "Reaction Count", "Reactions", "Likes"))
        comment_count = parse_int(first(row, "Comment Count", "Comments"))
        tags = split_values(first(row, "Tags", "Tag"))
        metadata = clean_metadata(
            {
                "post_url": post_url,
                "text": text,
                "saved_date": saved_at.isoformat(),
                "author": author,
                "author_profile_url": author_profile_url,
                "reaction_count": reaction_count,
                "comment_count": comment_count,
                "tags": tags,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="linkedin_saved_posts_csv",
            source_id=digest_source_id("linkedin_saved_posts_csv", post_url or text, "" if post_url else index),
            source_entity_type="saved_post",
            title=author or "LinkedIn saved post",
            content=self._content(text, author, post_url, reaction_count, comment_count),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["linkedin", *[tag.casefold() for tag in tags]],
            created_at=saved_at,
            updated_at=saved_at,
        )

    def _content(self, text: str, author: str, post_url: str, reaction_count: int | None, comment_count: int | None) -> str:
        parts = [part for part in (text, f"Author: {author}" if author else "", f"URL: {post_url}" if post_url else "") if part]
        if reaction_count is not None:
            parts.append(f"Reactions: {reaction_count}")
        if comment_count is not None:
            parts.append(f"Comments: {comment_count}")
        return "\n".join(parts)
