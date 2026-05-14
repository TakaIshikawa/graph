"""Adapter for Reddit saved items JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class RedditSavedJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "reddit_saved_json"

    @property
    def entity_types(self) -> list[str]:
        return ["post", "comment", "redditor"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types) if entity_types is not None else {"post", "comment"}
        if not requested.intersection(self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                items = self._read_items(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for item in items:
                unit = self._unit_from_item(item, path.name)
                if unit is None:
                    continue
                if sync_at and unit.created_at <= sync_at:
                    continue
                if unit.source_entity_type not in requested and "redditor" not in requested:
                    continue
                units.append(unit)

        item_units = sorted(units, key=lambda unit: (unit.created_at, unit.source_id))
        result.units.extend(unit for unit in item_units if unit.source_entity_type in requested)
        redditors = self._redditor_units(item_units) if "redditor" in requested else []
        result.units.extend(redditors)
        if requested.intersection({"post", "comment"}) and "redditor" in requested:
            result.edges.extend(self._redditor_edges(item_units, redditors, requested.intersection({"post", "comment"})))
        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id, edge.id))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.json") if child.is_file())

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._items(parsed)

    def _items(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [self._data(item) for item in value if self._data(item)]
        if not isinstance(value, dict):
            return []
        for key in ("saved", "items", "data", "children", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [self._data(item) for item in nested if self._data(item)]
            if isinstance(nested, dict):
                items = self._items(nested)
                if items:
                    return items
        item = self._data(value)
        return [item] if item else []

    def _data(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        data = item.get("data")
        if isinstance(data, dict):
            merged = dict(data)
            if "kind" in item and "kind" not in merged:
                merged["kind"] = item["kind"]
            return merged
        return item

    def _unit_from_item(self, item: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        body = self._first(item, "body", "selftext")
        link_title = self._first(item, "link_title")
        title = self._first(item, "title") or link_title
        entity_type = self._entity_type(item, body, title)
        if not title and not body:
            return None

        created_utc = self._parse_timestamp(item.get("created_utc") or item.get("created"))
        now = datetime.now(timezone.utc)
        permalink = self._first(item, "permalink")
        url = self._first(item, "url")
        metadata = {
            "id": self._first(item, "id"),
            "name": self._first(item, "name"),
            "title": title,
            "body": body,
            "selftext": self._first(item, "selftext"),
            "subreddit": self._first(item, "subreddit", "subreddit_name_prefixed"),
            "author": self._first(item, "author"),
            "permalink": self._absolute_permalink(permalink),
            "url": url,
            "created_utc": created_utc.isoformat() if created_utc else None,
            "score": self._parse_int(item.get("score")),
            "link_title": link_title,
            "source_file": source_file,
            "item": item,
        }
        tags = ["reddit", entity_type]
        if metadata["subreddit"]:
            tags.append(str(metadata["subreddit"]))
        unit_title = title or (f"Comment on {link_title}" if link_title else "Reddit saved comment")

        return KnowledgeUnit(
            source_project=SourceProject.REDDIT_SAVED_JSON,
            source_id=self._source_id(item, entity_type, title or body),
            source_entity_type=entity_type,
            title=unit_title,
            content=self._content(entity_type, title, body, link_title, metadata["permalink"], url),
            content_type=ContentType.ARTIFACT if entity_type == "post" else ContentType.INSIGHT,
            metadata=metadata,
            tags=tags,
            created_at=created_utc or now,
            updated_at=created_utc or now,
        )

    def _entity_type(self, item: dict[str, Any], body: str, title: str) -> str:
        kind = self._first(item, "kind")
        name = self._first(item, "name")
        if kind == "t1" or name.startswith("t1_") or (body and not title):
            return "comment"
        return "post"

    def _source_id(self, item: dict[str, Any], entity_type: str, fallback: str) -> str:
        explicit = self._first(item, "name") or self._first(item, "id")
        raw = explicit or "|".join([entity_type, fallback, str(item.get("created_utc") or "")])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"reddit_saved_json:{digest}"

    def _content(self, entity_type: str, title: str, body: str, link_title: str, permalink: str, url: str) -> str:
        parts = []
        if title:
            parts.append(title)
        if link_title and link_title != title:
            parts.append(f"Link title: {link_title}")
        if body:
            parts.append(body)
        if permalink:
            parts.append(f"Permalink: {permalink}")
        if url and (entity_type == "post" or url != permalink):
            parts.append(f"URL: {url}")
        return "\n\n".join(parts)

    def _redditor_units(self, items: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for item in items:
            author = str(item.metadata.get("author") or "").strip()
            key = self._normalized_author(author)
            if not key:
                continue
            grouped.setdefault(key, []).append(item)
            names.setdefault(key, author)

        units: list[KnowledgeUnit] = []
        for key, author_items in sorted(grouped.items()):
            unique_items = sorted({item.source_id: item for item in author_items}.values(), key=lambda item: item.source_id)
            author = names[key]
            post_count = sum(1 for item in unique_items if item.source_entity_type == "post")
            comment_count = sum(1 for item in unique_items if item.source_entity_type == "comment")
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.REDDIT_SAVED_JSON,
                    source_id=self._redditor_source_id(key),
                    source_entity_type="redditor",
                    title=author,
                    content=f"Reddit author: {author}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "author": author,
                        "normalized_author": key,
                        "saved_count": len(unique_items),
                        "post_count": post_count,
                        "comment_count": comment_count,
                        "subreddits": sorted({str(item.metadata.get("subreddit")) for item in unique_items if item.metadata.get("subreddit")}),
                        "first_saved_at": min(item.created_at for item in unique_items).isoformat(),
                        "last_saved_at": max(item.created_at for item in unique_items).isoformat(),
                        "item_source_ids": [item.source_id for item in unique_items],
                    },
                    tags=["reddit", "redditor"],
                    created_at=min(item.created_at for item in unique_items),
                    updated_at=max(item.updated_at for item in unique_items),
                )
            )
        return units

    def _redditor_edges(
        self,
        items: list[KnowledgeUnit],
        redditors: list[KnowledgeUnit],
        requested_item_types: set[str],
    ) -> list[KnowledgeEdge]:
        redditor_ids = {str(unit.metadata["normalized_author"]): unit.source_id for unit in redditors}
        edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            if item.source_entity_type not in requested_item_types:
                continue
            author = str(item.metadata.get("author") or "").strip()
            redditor_id = redditor_ids.get(self._normalized_author(author))
            if not redditor_id or (item.source_id, redditor_id) in seen:
                continue
            seen.add((item.source_id, redditor_id))
            edges.append(
                KnowledgeEdge(
                    id=self._redditor_edge_id(item.source_id, redditor_id),
                    from_unit_id=item.source_id,
                    to_unit_id=redditor_id,
                    relation=EdgeRelation.RELATES_TO,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.REDDIT_SAVED_JSON.value,
                        "from_entity_type": item.source_entity_type,
                        "to_entity_type": "redditor",
                        "relation_type": "saved_item_author",
                        "author": author,
                    },
                    created_at=item.created_at,
                )
            )
        return edges

    def _redditor_source_id(self, normalized_author: str) -> str:
        digest = hashlib.sha256(normalized_author.encode("utf-8")).hexdigest()[:24]
        return f"reddit_saved_json:redditor:{digest}"

    def _redditor_edge_id(self, item_source_id: str, redditor_source_id: str) -> str:
        digest = hashlib.sha256(f"{item_source_id}|{redditor_source_id}|saved_item_author".encode("utf-8")).hexdigest()[:24]
        return f"reddit_saved_json:edge:{digest}"

    def _normalized_author(self, author: str) -> str:
        return " ".join(author.casefold().split())

    def _absolute_permalink(self, value: str) -> str:
        if not value:
            return ""
        if value.startswith("http://") or value.startswith("https://"):
            return value
        if value.startswith("/"):
            return f"https://www.reddit.com{value}"
        return value

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is not None and not isinstance(value, (dict, list)) and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(float(str(value).strip()))
        except ValueError:
            return None

    def _parse_timestamp(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, TypeError, ValueError):
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
