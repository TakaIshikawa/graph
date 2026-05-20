"""Adapter for Reddit upvoted posts/comments CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class RedditUpvotedCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "reddit_upvoted_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["reddit_upvote"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "reddit_upvote" not in set(entity_types or self.entity_types):
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        units: dict[str, KnowledgeUnit] = {}
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows, start=1):
                unit = self._unit(row, path.name, index)
                if unit is None:
                    continue
                activity_at = self._activity_at(unit)
                if sync_at and activity_at and activity_at <= sync_at:
                    continue
                units.setdefault(unit.source_id, unit)

        result.units = sorted(units.values(), key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, row_index: int) -> KnowledgeUnit | None:
        title = first(row, "title", "post_title", "link_title", "submission_title", "name")
        body = first(row, "body", "comment_body", "selftext", "text", "content")
        permalink = self._absolute_permalink(first(row, "permalink", "link_permalink", "comments_url", "reddit_url"))
        url = first(row, "url", "link_url", "source_url", "external_url")
        thing_id = first(row, "thing", "thing_id", "fullname", "name", "id", "reddit_id")
        created_raw = first(row, "created_at", "created", "created_utc", "date_created", "timestamp")
        upvoted_raw = first(row, "upvoted_at", "upvote_at", "voted_at", "saved_at", "date", "upvoted_on")
        created_at = self._parse_datetime(created_raw)
        upvoted_at = self._parse_datetime(upvoted_raw)
        if not any([title, body, permalink, url, thing_id]):
            return None

        metadata = {
            "title": title,
            "body": body,
            "subreddit": first(row, "subreddit", "subreddit_name_prefixed", "community"),
            "author": first(row, "author", "username", "user"),
            "url": url,
            "permalink": permalink,
            "score": parse_int(first(row, "score", "ups", "upvotes", "points")),
            "created_at": created_at.isoformat() if created_at else created_raw,
            "upvoted_at": upvoted_at.isoformat() if upvoted_at else upvoted_raw,
            "thing_id": thing_id,
            "source_file": source_file,
            "row_index": row_index,
            "raw_record": dict(row),
        }
        now = datetime.now(timezone.utc)
        item_title = title or (f"Reddit comment by {metadata['author']}" if metadata["author"] else "Reddit upvote")
        return KnowledgeUnit(
            source_project="reddit_upvoted_csv",
            source_id=self._source_id(row, thing_id, permalink, url),
            source_entity_type="reddit_upvote",
            title=item_title,
            content=self._content(item_title, body, metadata),
            content_type=ContentType.INSIGHT,
            metadata=clean_metadata(metadata),
            tags=list(dict.fromkeys(tag for tag in ["reddit", "upvote", metadata["subreddit"]] if tag)),
            created_at=created_at or upvoted_at or now,
            updated_at=upvoted_at or created_at or now,
        )

    def _source_id(self, row: dict[str, Any], thing_id: str, permalink: str, url: str) -> str:
        stable = thing_id or permalink or url
        if stable:
            return digest_source_id("reddit_upvoted_csv", stable)
        return digest_source_id("reddit_upvoted_csv", dict(sorted((str(key), str(value)) for key, value in row.items())))

    def _content(self, title: str, body: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        if body and body != title:
            parts.append(body)
        for key, label in (
            ("subreddit", "Subreddit"),
            ("author", "Author"),
            ("score", "Score"),
            ("created_at", "Created"),
            ("upvoted_at", "Upvoted"),
            ("permalink", "Permalink"),
            ("url", "URL"),
        ):
            if metadata.get(key) not in ("", None, []):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(str(part) for part in parts if part not in ("", None))

    def _activity_at(self, unit: KnowledgeUnit) -> datetime | None:
        return self._parse_datetime(unit.metadata.get("upvoted_at")) or self._parse_datetime(unit.metadata.get("created_at"))

    def _absolute_permalink(self, value: str) -> str:
        if not value:
            return ""
        if value.startswith(("http://", "https://")):
            return value
        if value.startswith("/"):
            return f"https://www.reddit.com{value}"
        return value

    def _parse_datetime(self, value: Any) -> datetime | None:
        text = "" if value is None else str(value).strip()
        if not text:
            return None
        try:
            return datetime.fromtimestamp(float(text), tz=timezone.utc)
        except (OSError, OverflowError, TypeError, ValueError):
            pass
        for candidate in (text, text.replace("Z", "+00:00"), f"{text}T00:00:00"):
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                continue
            return ensure_utc(parsed)
        for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None
