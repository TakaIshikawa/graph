"""Adapter for Hacker News submitted stories CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class HackerNewsSubmissionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "hacker_news_submissions_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["submission"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "submission" not in entity_types:
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
        item_id = first(row, "ID", "Id", "Item ID", "Item Id", "HN ID", "Hacker News ID")
        title = first(row, "Title", "Story Title", "Name")
        url = first(row, "URL", "Url", "Link")
        text = first(row, "Text", "Body", "Contents")
        submitted_text = first(row, "Submitted At", "Created At", "Created", "Time", "Date")
        submitted_at = parse_datetime(submitted_text)
        score = parse_int(first(row, "Score", "Points"))
        comments = parse_int(first(row, "Comments", "Comment Count", "Descendants"))
        item_type = first(row, "Type", "Item Type") or ("story" if url else "text")
        hn_item_url = self._hn_item_url(item_id)
        source_url = url or hn_item_url

        if not any([item_id, title, url, text, submitted_text]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "item_id": parse_int(item_id) if item_id else None,
                "hn_item_id": parse_int(item_id) if item_id else None,
                "title": title,
                "url": source_url,
                "external_url": url,
                "source_url": source_url,
                "hn_item_url": hn_item_url,
                "submitted_at": submitted_at.isoformat() if submitted_at else submitted_text,
                "score": score,
                "comment_count": comments,
                "comments": comments,
                "item_type": item_type,
                "hn_item_type": item_type,
                "domain": self._domain(url),
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="hacker_news_submissions_csv",
            source_id=self._source_id(item_id, url, title, index),
            source_entity_type="submission",
            title=title or url or (f"Hacker News item {item_id}" if item_id else "Hacker News submission"),
            content=self._content(title, text, url, hn_item_url, score, comments, submitted_at or submitted_text),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=clean_metadata({"base": ["hacker_news", "submitted", item_type]}).get("base", []),
            created_at=submitted_at or now,
            updated_at=submitted_at or now,
        )

    def _source_id(self, item_id: str, url: str, title: str, index: int) -> str:
        if item_id:
            return f"hacker_news_submissions_csv:submission:{item_id}"
        return digest_source_id("hacker_news_submissions_csv:submission", url, title, index if not (url or title) else "")

    def _hn_item_url(self, item_id: str) -> str:
        return f"https://news.ycombinator.com/item?id={item_id}" if item_id else ""

    def _domain(self, url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        return parsed.netloc.lower().removeprefix("www.")

    def _content(
        self,
        title: str,
        text: str,
        url: str,
        hn_item_url: str,
        score: int | None,
        comments: int | None,
        submitted_at: datetime | str | None,
    ) -> str:
        parts = [part for part in (title, text, f"URL: {url}" if url else "", f"Hacker News: {hn_item_url}" if hn_item_url else "") if part]
        if score is not None:
            parts.append(f"Score: {score}")
        if comments is not None:
            parts.append(f"Comments: {comments}")
        if submitted_at:
            parts.append(f"Submitted At: {submitted_at.isoformat() if isinstance(submitted_at, datetime) else submitted_at}")
        return "\n".join(parts)

    def _dedupe_key(self, unit: KnowledgeUnit) -> tuple[datetime, str, str]:
        return (unit.updated_at, str(unit.metadata.get("source_file") or ""), unit.title)
