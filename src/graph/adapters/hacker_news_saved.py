"""Adapter for Hacker News saved/upvoted item JSON exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class HackerNewsSavedAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "hacker_news_saved"

    @property
    def entity_types(self) -> list[str]:
        return ["saved_item", "submitter"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types or self.entity_types)
        if not requested.intersection(self.entity_types):
            return result

        sync_at = self._sync_datetime(since) if since else None
        saved_items: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                items = self._read_items(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue

            for item in items:
                unit = self._unit_from_item(item, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                saved_items.append(unit)

        submitter_units = self._submitter_units(saved_items)
        if "saved_item" in requested:
            result.units.extend(saved_items)
        if "submitter" in requested:
            result.units.extend(submitter_units)
        if {"saved_item", "submitter"}.issubset(requested):
            result.edges.extend(self._submitter_edges(submitter_units, saved_items))
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        path = Path(self.path).expanduser()
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(child for child in path.rglob("*.json") if child.is_file())
        return []

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("items", "saved_items"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    def _unit_from_item(self, item: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        item_id = self._first(item, "id", "item_id")
        url = self._first(item, "url")
        hn_item_id = self._hn_item_id(item_id, url)
        hn_item_url = self._hn_item_url(str(hn_item_id) if hn_item_id is not None else item_id)
        source_url = url or hn_item_url
        title = self._first(item, "title") or url or (f"Hacker News item {item_id}" if item_id else "")
        text = self._first(item, "text")
        if not title and not text:
            return None

        item_time = self._parse_unix_time(item.get("time"))
        item_time_iso = item_time.isoformat() if item_time else None
        now = datetime.now(timezone.utc)
        kids = item.get("kids")
        comment_count = len(kids) if isinstance(kids, list) else 0
        item_type = self._normalized_item_type(item, url, title, text)
        source_id = self._source_id(item_id, source_url, title)
        parent_id = self._first_int(item, "parent", "parent_id", "parentId")
        story_id = self._first_int(item, "story_id", "story", "storyId", "root_id", "root")
        submitter = self._first(item, "author", "by", "submitter")
        metadata: dict[str, Any] = {
            "item_id": self._parse_int(item_id),
            "hn_item_id": hn_item_id,
            "author": submitter,
            "submitter": submitter,
            "score": self._parse_int(item.get("score")),
            "item_type": item_type,
            "hn_item_type": item_type,
            "comment_count": comment_count,
            "time": self._parse_int(item.get("time")),
            "time_iso": item_time_iso,
            "source_url": source_url,
            "hn_item_url": hn_item_url,
            "domain": self._domain(url),
            "source_file": source_file,
        }
        if parent_id is not None:
            metadata["parent_id"] = parent_id
            metadata["hn_parent_id"] = parent_id
        if story_id is not None:
            metadata["story_id"] = story_id
            metadata["hn_story_id"] = story_id
        if url:
            metadata["external_url"] = url
        if isinstance(kids, list):
            metadata["kids"] = kids

        return KnowledgeUnit(
            source_project=SourceProject.HACKER_NEWS_SAVED,
            source_id=source_id,
            source_entity_type="saved_item",
            title=title,
            content=self._content(title, text, url, hn_item_url),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["hacker_news", item_type],
            created_at=item_time or now,
            updated_at=item_time or now,
        )

    def _content(self, title: str, text: str, url: str, hn_item_url: str) -> str:
        parts = [title]
        if text:
            parts.append(text)
        if url:
            parts.append(f"URL: {url}")
        if hn_item_url:
            parts.append(f"Hacker News: {hn_item_url}")
        return "\n".join(parts)

    def _source_id(self, item_id: str, source_url: str, title: str) -> str:
        if item_id:
            return f"hacker_news_saved:{item_id}"
        digest = hashlib.sha256((source_url or title).encode("utf-8")).hexdigest()[:24]
        return f"hacker_news_saved:{digest}"

    def _submitter_units(self, saved_items: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for item in saved_items:
            submitter = str(item.metadata.get("submitter") or "")
            key = self._submitter_key(submitter)
            if not key:
                continue
            grouped.setdefault(key, []).append(item)
            names.setdefault(key, submitter.strip())

        units: list[KnowledgeUnit] = []
        for key, items in sorted(grouped.items()):
            unique_items = sorted({item.source_id: item for item in items}.values(), key=lambda item: item.source_id)
            submitter = names[key]
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.HACKER_NEWS_SAVED,
                    source_id=self._submitter_source_id(key),
                    source_entity_type="submitter",
                    title=submitter,
                    content=f"Hacker News submitter: {submitter}\nItems: {len(unique_items)}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "submitter": submitter,
                        "normalized_submitter": key,
                        "item_count": len(unique_items),
                        "item_source_ids": [item.source_id for item in unique_items],
                        "item_types": sorted({str(item.metadata.get("item_type")) for item in unique_items if item.metadata.get("item_type")}),
                        "domains": sorted({str(item.metadata.get("domain")) for item in unique_items if item.metadata.get("domain")}),
                        "source_files": sorted({str(item.metadata.get("source_file")) for item in unique_items if item.metadata.get("source_file")}),
                    },
                    tags=["hacker_news", "submitter"],
                    created_at=min(item.created_at for item in unique_items),
                    updated_at=max(item.updated_at for item in unique_items),
                )
            )
        return units

    def _submitter_edges(self, submitters: list[KnowledgeUnit], saved_items: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        submitter_ids = {str(submitter.metadata.get("normalized_submitter")): submitter.source_id for submitter in submitters}
        edges: list[KnowledgeEdge] = []
        for item in saved_items:
            key = self._submitter_key(str(item.metadata.get("submitter") or ""))
            submitter_id = submitter_ids.get(key)
            if not submitter_id:
                continue
            digest = hashlib.sha256(f"{submitter_id}|{item.source_id}|contains".encode("utf-8")).hexdigest()[:24]
            edges.append(
                KnowledgeEdge(
                    id=f"hacker-news-saved-submitter-contains-{digest}",
                    from_unit_id=submitter_id,
                    to_unit_id=item.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.HACKER_NEWS_SAVED.value,
                        "from_entity_type": "submitter",
                        "to_entity_type": "saved_item",
                        "submitter": item.metadata.get("submitter"),
                    },
                    created_at=item.created_at,
                )
            )
        return edges

    def _submitter_key(self, submitter: str) -> str:
        return " ".join(submitter.casefold().split())

    def _submitter_source_id(self, normalized_submitter: str) -> str:
        digest = hashlib.sha256(normalized_submitter.encode("utf-8")).hexdigest()[:24]
        return f"hacker_news_saved:submitter:{digest}"

    def _hn_item_url(self, item_id: str) -> str:
        if not item_id:
            return ""
        return f"https://news.ycombinator.com/item?id={item_id}"

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _first_int(self, item: dict[str, Any], *keys: str) -> int | None:
        for key in keys:
            parsed = self._parse_int(item.get(key))
            if parsed is not None:
                return parsed
        return None

    def _hn_item_id(self, item_id: str, url: str) -> int | None:
        parsed = self._parse_int(item_id)
        if parsed is not None:
            return parsed
        match = re.search(r"[?&]id=(\d+)", url)
        return int(match.group(1)) if match else None

    def _normalized_item_type(self, item: dict[str, Any], url: str, title: str, text: str) -> str:
        raw_type = self._first(item, "type", "item_type", "hn_item_type").casefold()
        aliases = {
            "story": "story",
            "comment": "comment",
            "job": "job",
            "poll": "poll",
            "ask": "story",
            "show": "story",
        }
        if raw_type in aliases:
            return aliases[raw_type]
        if raw_type:
            return "unknown"
        if self._first(item, "parent", "parent_id", "parentId"):
            return "comment"
        if text and not title and not url:
            return "comment"
        if title or url:
            return "story"
        return "unknown"

    def _domain(self, url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        if not parsed.hostname and "://" not in url:
            parsed = urlparse(f"https://{url}")
        host = (parsed.hostname or "").rstrip(".").casefold()
        if not host or any(char.isspace() for char in host):
            return ""
        return host.removeprefix("www.")

    def _parse_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_unix_time(self, value: Any) -> datetime | None:
        parsed = self._parse_int(value)
        if parsed is None:
            return None
        try:
            return datetime.fromtimestamp(parsed, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
