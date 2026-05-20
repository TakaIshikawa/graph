"""Adapter for Product Hunt votes JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime, parse_int
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class ProductHuntVotesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "product_hunt_votes_json"

    @property
    def entity_types(self) -> list[str]:
        return ["product_vote"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "product_vote" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        units_by_id: dict[str, KnowledgeUnit] = {}

        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit(record, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                previous = units_by_id.get(unit.source_id)
                if previous is None or self._dedupe_key(unit) > self._dedupe_key(previous):
                    units_by_id[unit.source_id] = unit

        result.units = sorted(units_by_id.values(), key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("votes", "upvotes", "products", "items", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        product = record.get("product") if isinstance(record.get("product"), dict) else {}
        name = self._first(record, product, "name", "product_name", "title")
        tagline = self._first(record, product, "tagline", "product_tagline", "subtitle")
        url = self._first(record, product, "url", "product_url", "discussion_url", "comments_url", "website", "redirect_url")
        makers = self._names(record.get("makers") or record.get("maker_names") or product.get("makers") or product.get("maker_names"))
        topics = self._names(record.get("topics") or record.get("tags") or product.get("topics") or product.get("tags"))
        voted_text = self._first(record, product, "voted_at", "upvoted_at", "vote_timestamp", "votedAt", "upvotedAt", "created_at", "createdAt")
        featured_text = self._first(record, product, "featured_at", "featuredAt", "featured_date")
        voted_at = parse_datetime(voted_text)
        featured_at = parse_datetime(featured_text)
        votes_count = parse_int(self._first(record, product, "votes_count", "vote_count", "votes", "upvotes", "upvotes_count"))
        comments_count = parse_int(self._first(record, product, "comments_count", "comment_count", "comments"))
        vote_id = self._first(record, {}, "vote_id", "upvote_id", "id")
        product_id = self._first(product, record, "product_id", "post_id", "id")

        if not any([name, url, vote_id, product_id]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "name": name,
                "tagline": tagline,
                "url": url,
                "makers": makers,
                "topics": topics,
                "voted_at": voted_at.isoformat() if voted_at else voted_text,
                "featured_at": featured_at.isoformat() if featured_at else featured_text,
                "votes_count": votes_count,
                "comments_count": comments_count,
                "vote_id": parse_int(vote_id) if vote_id else None,
                "product_id": parse_int(product_id) if product_id else None,
                "source_file": source_file,
                "record": record,
            }
        )
        return KnowledgeUnit(
            source_project="product_hunt_votes_json",
            source_id=self._source_id(vote_id, product_id, url, name, voted_at or voted_text, source_file, index),
            source_entity_type="product_vote",
            title=name or url or f"Product Hunt vote {vote_id or product_id}",
            content=self._content(name, tagline, url, makers, voted_at or voted_text, featured_at or featured_text, votes_count, comments_count),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(["producthunt", "product_vote", *topics])),
            created_at=featured_at or voted_at or now,
            updated_at=voted_at or featured_at or now,
        )

    def _source_id(self, vote_id: str, product_id: str, url: str, name: str, voted_at: datetime | str | None, source_file: str, index: int) -> str:
        if vote_id:
            return f"product_hunt_votes_json:product_vote:{vote_id}"
        if product_id and voted_at:
            return f"product_hunt_votes_json:product_vote:{product_id}:{voted_at.isoformat() if isinstance(voted_at, datetime) else voted_at}"
        return digest_source_id("product_hunt_votes_json:product_vote", url, name, voted_at, source_file, index if not (url or name or voted_at) else "")

    def _content(
        self,
        name: str,
        tagline: str,
        url: str,
        makers: list[str],
        voted_at: datetime | str | None,
        featured_at: datetime | str | None,
        votes_count: int | None,
        comments_count: int | None,
    ) -> str:
        parts = [name, tagline, f"URL: {url}" if url else ""]
        if makers:
            parts.append(f"Makers: {', '.join(makers)}")
        if voted_at:
            parts.append(f"Voted At: {voted_at.isoformat() if isinstance(voted_at, datetime) else voted_at}")
        if featured_at:
            parts.append(f"Featured At: {featured_at.isoformat() if isinstance(featured_at, datetime) else featured_at}")
        if votes_count is not None:
            parts.append(f"Votes: {votes_count}")
        if comments_count is not None:
            parts.append(f"Comments: {comments_count}")
        return "\n".join(part for part in parts if part)

    def _first(self, primary: dict[str, Any], secondary: dict[str, Any], *keys: str) -> str:
        for source in (primary, secondary):
            for key in keys:
                value = source.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
        return ""

    def _names(self, value: Any) -> list[str]:
        if isinstance(value, list):
            names = [self._name(item) for item in value]
            return [name for name in names if name]
        if isinstance(value, str):
            return [part.strip() for part in value.replace(";", ",").replace("|", ",").split(",") if part.strip()]
        if isinstance(value, dict):
            name = self._name(value)
            return [name] if name else []
        return []

    def _name(self, value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("name") or value.get("title") or value.get("username")
        return "" if value is None else str(value).strip()

    def _dedupe_key(self, unit: KnowledgeUnit) -> tuple[datetime, datetime, str, str]:
        return (unit.updated_at, unit.created_at, str(unit.metadata.get("source_file") or ""), unit.title)
