"""Adapter for Matter reader app article and highlight exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class MatterAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "matter"

    @property
    def entity_types(self) -> list[str]:
        return ["article", "highlight"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and not any(et in entity_types for et in ["article", "highlight"]):
            return result

        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists() or not path.is_file():
            return result

        try:
            items = self._read_items(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return result

        sync_at = self._sync_datetime(since) if since else None
        for item in items:
            url = self._url(item)
            title = self._title(item, url)
            if not url and not title:
                continue

            created_text = self._first(item, "created_at", "created", "date_created", "added_at")
            updated_text = self._first(item, "updated_at", "updated", "date_updated", "modified_at")
            created_at = self._parse_datetime(created_text)
            updated_at = self._parse_datetime(updated_text)
            comparable_at = updated_at or created_at
            if sync_at and comparable_at and comparable_at <= sync_at:
                continue

            content = self._first(item, "content", "body", "html")
            author = self._first(item, "author", "author_name")
            tags = self._parse_tags(item.get("tags") or item.get("tag"))
            reading_progress = self._first(item, "reading_progress", "progress", "completion")
            matter_id = self._first(item, "id", "item_id", "article_id")

            # Check if this item has embedded highlights
            highlights = self._extract_highlights(item)

            # Create article unit
            if not entity_types or "article" in entity_types:
                article_content = self._content(title, url, author, content, highlights)
                result.units.append(
                    KnowledgeUnit(
                        source_project=SourceProject.MATTER,
                        source_id=self._source_id(matter_id, url, title),
                        source_entity_type="article",
                        title=title or url or "Untitled Matter article",
                        content=article_content,
                        content_type=ContentType.ARTIFACT,
                        metadata={
                            "url": url,
                            "author": author,
                            "content": content,
                            "reading_progress": reading_progress,
                            "tags": tags,
                            "created_at": created_text,
                            "updated_at": updated_text,
                            "matter_id": matter_id,
                            "highlight_count": str(len(highlights)) if highlights else "",
                        },
                        tags=tags,
                        created_at=created_at or updated_at or datetime.now(timezone.utc),
                        updated_at=updated_at or created_at or datetime.now(timezone.utc),
                    )
                )

            # Create separate highlight units if requested
            if highlights and (not entity_types or "highlight" in entity_types):
                for idx, highlight in enumerate(highlights):
                    highlight_text = highlight.get("text") or highlight.get("highlight") or ""
                    note = highlight.get("note") or highlight.get("annotation") or ""
                    if not highlight_text:
                        continue

                    highlight_id = highlight.get("id") or f"{matter_id}_hl_{idx}"
                    highlight_created = self._parse_datetime(
                        highlight.get("created_at") or highlight.get("created") or created_text
                    )

                    result.units.append(
                        KnowledgeUnit(
                            source_project=SourceProject.MATTER,
                            source_id=f"matter_highlight:{highlight_id}",
                            source_entity_type="highlight",
                            title=f"Highlight from {title}" if title else "Matter highlight",
                            content=self._highlight_content(highlight_text, note, title, url),
                            content_type=ContentType.ARTIFACT,
                            metadata={
                                "highlight": highlight_text,
                                "note": note,
                                "source_url": url,
                                "source_title": title,
                                "source_id": matter_id,
                                "created_at": highlight.get("created_at") or highlight.get("created") or created_text,
                            },
                            tags=tags,
                            created_at=highlight_created or created_at or datetime.now(timezone.utc),
                            updated_at=highlight_created or created_at or datetime.now(timezone.utc),
                        )
                    )

        return result

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._json_items(parsed)

    def _json_items(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []

        for key in ("articles", "items", "entries", "list", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                return [item for item in nested.values() if isinstance(item, dict)]

        if any(isinstance(item, dict) for item in value.values()):
            return [item for item in value.values() if isinstance(item, dict)]
        return [value]

    def _extract_highlights(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        highlights_data = item.get("highlights") or item.get("annotations")
        if not highlights_data:
            return []

        if isinstance(highlights_data, list):
            return [h for h in highlights_data if isinstance(h, dict)]
        elif isinstance(highlights_data, dict):
            return [h for h in highlights_data.values() if isinstance(h, dict)]
        return []

    def _title(self, item: dict[str, Any], url: str) -> str:
        return self._first(item, "title", "article_title", "name") or url

    def _url(self, item: dict[str, Any]) -> str:
        return self._first(item, "url", "href", "link")

    def _source_id(self, matter_id: str, url: str, title: str) -> str:
        if matter_id:
            return f"matter:{matter_id}"
        if url:
            return f"url:{url}"
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()
        return f"matter:{digest[:24]}"

    def _content(
        self,
        title: str,
        url: str,
        author: str,
        content: str,
        highlights: list[dict[str, Any]],
    ) -> str:
        parts = []
        if title:
            parts.append(title)
        if author:
            parts.append(f"Author: {author}")
        if url:
            parts.append(f"URL: {url}")
        if content:
            # Truncate very long content
            if len(content) > 5000:
                content = content[:5000] + "..."
            parts.append(f"Content: {content}")
        if highlights:
            parts.append(f"\nHighlights ({len(highlights)}):")
            for idx, h in enumerate(highlights[:10], 1):  # Limit to first 10 highlights
                text = h.get("text") or h.get("highlight") or ""
                if text:
                    parts.append(f"{idx}. {text}")
        return "\n".join(parts)

    def _highlight_content(self, highlight: str, note: str, source_title: str, source_url: str) -> str:
        parts = [f"Highlight: {highlight}"]
        if note:
            parts.append(f"Note: {note}")
        if source_title:
            parts.append(f"Source: {source_title}")
        if source_url:
            parts.append(f"URL: {source_url}")
        return "\n".join(parts)

    def _parse_tags(self, value: Any) -> list[str]:
        raw_tags: list[Any]
        if isinstance(value, dict):
            raw_tags = []
            for key, tag_value in value.items():
                if isinstance(tag_value, dict):
                    raw_tags.append(tag_value.get("tag") or tag_value.get("name") or key)
                else:
                    raw_tags.append(tag_value or key)
        elif isinstance(value, list):
            raw_tags = value
        elif isinstance(value, str):
            raw_tags = re.split(r"[,;|]", value)
        else:
            raw_tags = []

        tags: list[str] = []
        for tag in raw_tags:
            if isinstance(tag, dict):
                tag = tag.get("tag") or tag.get("name") or ""
            normalized = re.sub(r"\s+", " ", str(tag).strip().removeprefix("#")).strip().lower()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        if re.fullmatch(r"\d+(?:\.0+)?", value):
            try:
                return datetime.fromtimestamp(int(float(value)), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
