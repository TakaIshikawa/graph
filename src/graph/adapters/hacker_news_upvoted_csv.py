"""Adapter for Hacker News upvoted item CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class HackerNewsUpvotedCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "hacker_news_upvoted_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["upvoted_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "upvoted_item" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        units_by_id: dict[str, KnowledgeUnit] = {}

        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                previous = units_by_id.get(unit.source_id)
                if previous is None or self._dedupe_key(unit) > self._dedupe_key(previous):
                    units_by_id[unit.source_id] = unit

        result.units = sorted(units_by_id.values(), key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Title", "Story Title", "Name")
        url = first(row, "URL", "Url", "Link")
        item_id = first(row, "Item ID", "Item Id", "ID", "Id", "HN ID", "Hacker News ID")
        item_type = first(row, "Type", "Item Type") or "unknown"
        author = first(row, "Author", "By", "Submitter", "User")
        score = parse_int(first(row, "Score", "Points"))
        comments = parse_int(first(row, "Comments", "Comment Count", "Descendants"))
        created_text = first(row, "Created At", "Created", "Time", "Submitted At")
        upvoted_text = first(row, "Upvoted At", "Upvote At", "Voted At", "Date")
        created_at = parse_datetime(created_text)
        upvoted_at = parse_datetime(upvoted_text)
        hn_item_url = self._hn_item_url(item_id)

        if not any([title, url, item_id, author, created_text, upvoted_text]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "title": title,
                "url": url,
                "source_url": url or hn_item_url,
                "external_url": url,
                "item_id": parse_int(item_id) if item_id else None,
                "hn_item_id": parse_int(item_id) if item_id else None,
                "hn_item_url": hn_item_url,
                "item_type": item_type,
                "hn_item_type": item_type,
                "author": author,
                "submitter": author,
                "score": score,
                "comment_count": comments,
                "comments": comments,
                "created_at": created_at.isoformat() if created_at else created_text,
                "upvoted_at": upvoted_at.isoformat() if upvoted_at else upvoted_text,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="hacker_news_upvoted_csv",
            source_id=self._source_id(item_id, url, title, index),
            source_entity_type="upvoted_item",
            title=title or url or (f"Hacker News item {item_id}" if item_id else "Hacker News upvoted item"),
            content=self._content(title, url, hn_item_url, item_type, author, score, comments, upvoted_at or upvoted_text),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["hacker_news", item_type],
            created_at=created_at or upvoted_at or now,
            updated_at=upvoted_at or created_at or now,
        )

    def _source_id(self, item_id: str, url: str, title: str, index: int) -> str:
        if item_id:
            return f"hacker_news_upvoted_csv:upvoted_item:{item_id}"
        return digest_source_id("hacker_news_upvoted_csv:upvoted_item", url, title, index if not (url or title) else "")

    def _hn_item_url(self, item_id: str) -> str:
        if not item_id:
            return ""
        return f"https://news.ycombinator.com/item?id={item_id}"

    def _content(
        self,
        title: str,
        url: str,
        hn_item_url: str,
        item_type: str,
        author: str,
        score: int | None,
        comments: int | None,
        upvoted_at: datetime | str | None,
    ) -> str:
        parts = [part for part in (title, f"URL: {url}" if url else "", f"Hacker News: {hn_item_url}" if hn_item_url else "") if part]
        if item_type:
            parts.append(f"Type: {item_type}")
        if author:
            parts.append(f"Author: {author}")
        if score is not None:
            parts.append(f"Score: {score}")
        if comments is not None:
            parts.append(f"Comments: {comments}")
        if upvoted_at:
            parts.append(f"Upvoted At: {upvoted_at.isoformat() if isinstance(upvoted_at, datetime) else upvoted_at}")
        return "\n".join(parts)

    def _dedupe_key(self, unit: KnowledgeUnit) -> tuple[datetime, datetime, str, str]:
        return (unit.updated_at, unit.created_at, str(unit.metadata.get("source_file") or ""), unit.title)
