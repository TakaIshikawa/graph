"""Adapter for Omnivore JSON exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class OmnivoreJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "omnivore_json"

    @property
    def entity_types(self) -> list[str]:
        return ["article", "highlight", "note", "label"]

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

        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists() or not path.is_file():
            return result

        try:
            items = self._read_items(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return result

        sync_at = self._sync_datetime(since) if since else None
        emitted_edges: set[tuple[str, str]] = set()
        label_links: dict[str, dict[str, Any]] = {}
        now = datetime.now(timezone.utc)

        for item in items:
            article = self._article_record(item)
            if article is None:
                continue

            url = self._first(article, "url", "originalUrl", "original_url", "canonicalUrl")
            title = self._first(article, "title", "name") or url
            if not title and not url:
                continue

            omnivore_id = self._first(article, "id", "itemId", "item_id", "pageId", "page_id")
            article_source_id = self._article_source_id(omnivore_id, url, title)
            saved_text = self._first(
                article,
                "savedAt",
                "saved_at",
                "createdAt",
                "created_at",
                "dateSaved",
                "date_saved",
            )
            read_text = self._first(
                article,
                "readAt",
                "read_at",
                "readingProgressLastReadAt",
                "reading_progress_last_read_at",
            )
            updated_text = self._first(article, "updatedAt", "updated_at", "modifiedAt")
            saved_at = self._parse_datetime(saved_text)
            read_at = self._parse_datetime(read_text)
            updated_at = self._parse_datetime(updated_text)
            comparable_at = updated_at or read_at or saved_at
            if sync_at and comparable_at and comparable_at <= sync_at:
                continue

            labels = self._labels(article)
            article_label_entries = self._label_entries(article)
            author = self._first(article, "author", "authorName", "author_name", "byline")
            state = self._first(article, "state", "status")
            highlights = self._highlights(item, article)
            if "label" in requested:
                self._add_label_source_link(label_links, article_label_entries, article_source_id, "article")

            if "article" in requested:
                article_unit = KnowledgeUnit(
                    source_project=SourceProject.OMNIVORE_JSON,
                    source_id=article_source_id,
                    source_entity_type="article",
                    title=title or "Untitled Omnivore article",
                    content=self._article_content(title, url, author, labels, highlights),
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "omnivore_id": omnivore_id,
                        "url": url,
                        "author": author,
                        "state": state,
                        "labels": labels,
                        "saved_at": saved_text,
                        "read_at": read_text,
                        "archived": self._is_archived(article, state),
                        "read": self._is_read(article, state, read_text),
                        "highlight_count": len(highlights),
                    },
                    tags=labels,
                    created_at=saved_at or read_at or updated_at or now,
                    updated_at=updated_at or read_at or saved_at or now,
                )
                result.units.append(article_unit)

            if requested.intersection({"highlight", "note"}) or "label" in requested:
                for index, highlight in enumerate(highlights):
                    child_label_entries = self._merge_label_entries(article_label_entries, self._label_entries(highlight))
                    child_labels = [entry[0] for entry in child_label_entries]
                    child_unit = self._highlight_unit(
                        highlight,
                        index,
                        article_source_id,
                        omnivore_id,
                        title,
                        url,
                        child_labels,
                        saved_at or updated_at or now,
                    )
                    if child_unit is None:
                        continue
                    if "label" in requested and child_unit.source_entity_type == "highlight":
                        self._add_label_source_link(label_links, child_label_entries, child_unit.source_id, "highlight")
                    if child_unit.source_entity_type not in requested:
                        continue

                    result.units.append(child_unit)
                    edge_key = (article_source_id, child_unit.source_id)
                    if edge_key not in emitted_edges:
                        emitted_edges.add(edge_key)
                        result.edges.append(
                            KnowledgeEdge(
                                id=self._edge_id(article_source_id, child_unit.source_id),
                                from_unit_id=article_source_id,
                                to_unit_id=child_unit.source_id,
                                relation=EdgeRelation.CONTAINS,
                                source=EdgeSource.SOURCE,
                                metadata={
                                    "source_project": SourceProject.OMNIVORE_JSON.value,
                                    "from_entity_type": "article",
                                    "to_entity_type": child_unit.source_entity_type,
                                    "relation_type": "article_highlight",
                                },
                            )
                        )

        if "label" in requested:
            label_units = self._label_units(label_links, now)
            result.units.extend(label_units)
            requested_content_types = requested.intersection({"article", "highlight"})
            if requested_content_types:
                result.edges.extend(self._label_edges(label_units, label_links, requested_content_types))

        result.units.sort(key=lambda unit: (unit.source_entity_type, unit.source_id))
        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id))
        return result

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._json_items(parsed)

    def _json_items(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []

        for key in ("items", "articles", "pages", "entries", "results", "data"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                nested_items = self._json_items(nested)
                if nested_items:
                    return nested_items

        if isinstance(value.get("page"), dict) or self._looks_like_article(value):
            return [value]

        return [
            item
            for item in value.values()
            if isinstance(item, dict) and self._looks_like_article(item)
        ]

    def _article_record(self, item: dict[str, Any]) -> dict[str, Any] | None:
        page = item.get("page")
        if isinstance(page, dict):
            merged = dict(page)
            for key in (
                "highlights",
                "labels",
                "state",
                "savedAt",
                "saved_at",
                "readAt",
                "read_at",
            ):
                if key in item and key not in merged:
                    merged[key] = item[key]
            return merged
        if self._looks_like_article(item):
            return item
        return None

    def _looks_like_article(self, item: dict[str, Any]) -> bool:
        return any(
            key in item
            for key in ("url", "originalUrl", "original_url", "canonicalUrl", "title")
        )

    def _highlights(self, item: dict[str, Any], article: dict[str, Any]) -> list[dict[str, Any]]:
        raw = (
            item.get("highlights")
            or article.get("highlights")
            or item.get("annotations")
            or article.get("annotations")
        )
        if isinstance(raw, list):
            return [highlight for highlight in raw if isinstance(highlight, dict)]
        if isinstance(raw, dict):
            return [highlight for highlight in raw.values() if isinstance(highlight, dict)]
        return []

    def _highlight_unit(
        self,
        highlight: dict[str, Any],
        index: int,
        article_source_id: str,
        omnivore_id: str,
        article_title: str,
        url: str,
        labels: list[str],
        fallback_time: datetime,
    ) -> KnowledgeUnit | None:
        text = self._first(highlight, "quote", "text", "highlight", "highlightText")
        note = self._first(highlight, "annotation", "note", "notes", "comment")
        if not text and not note:
            return None

        highlight_id = self._first(highlight, "id", "highlightId", "highlight_id")
        source_id = self._highlight_source_id(highlight_id, article_source_id, text, note, index)
        created_text = self._first(highlight, "createdAt", "created_at", "highlightedAt", "updatedAt")
        created_at = self._parse_datetime(created_text) or fallback_time
        entity_type = "highlight" if text else "note"

        return KnowledgeUnit(
            source_project=SourceProject.OMNIVORE_JSON,
            source_id=source_id,
            source_entity_type=entity_type,
            title=(
                f"{'Highlight' if text else 'Note'} from {article_title}"
                if article_title
                else "Omnivore highlight"
            ),
            content=self._highlight_content(text, note, article_title, url),
            content_type=ContentType.INSIGHT,
            metadata={
                "omnivore_id": highlight_id,
                "article_omnivore_id": omnivore_id,
                "article_source_id": article_source_id,
                "source_title": article_title,
                "source_url": url,
                "text": text,
                "note": note,
                "color": self._first(highlight, "color"),
                "prefix": self._first(highlight, "prefix"),
                "suffix": self._first(highlight, "suffix"),
                "created_at": created_text,
            },
            tags=labels,
            created_at=created_at,
            updated_at=created_at,
        )

    def _article_source_id(self, omnivore_id: str, url: str, title: str) -> str:
        if omnivore_id:
            return f"omnivore:{omnivore_id}"
        if url:
            return f"url:{url}"
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()
        return f"omnivore:{digest[:24]}"

    def _highlight_source_id(
        self,
        highlight_id: str,
        article_source_id: str,
        text: str,
        note: str,
        index: int,
    ) -> str:
        if highlight_id:
            return f"omnivore_highlight:{highlight_id}"
        digest = hashlib.sha256(
            "\n".join([article_source_id, text, note, str(index)]).encode("utf-8")
        ).hexdigest()
        return f"omnivore_highlight:{digest[:24]}"

    def _edge_id(self, parent_source_id: str, child_source_id: str) -> str:
        digest = hashlib.sha256(
            f"{parent_source_id}\0{child_source_id}".encode("utf-8")
        ).hexdigest()
        return f"omnivore_edge:{digest[:24]}"

    def _article_content(
        self,
        title: str,
        url: str,
        author: str,
        labels: list[str],
        highlights: list[dict[str, Any]],
    ) -> str:
        parts = [
            part
            for part in (
                title,
                f"Author: {author}" if author else "",
                f"URL: {url}" if url else "",
            )
            if part
        ]
        if labels:
            parts.append(f"Labels: {', '.join(labels)}")
        if highlights:
            parts.append(f"Highlights: {len(highlights)}")
        return "\n".join(parts)

    def _highlight_content(self, text: str, note: str, article_title: str, url: str) -> str:
        parts: list[str] = []
        if text:
            parts.append(text)
        if note:
            parts.append(f"Note: {note}")
        if article_title:
            parts.append(f"Source: {article_title}")
        if url:
            parts.append(f"URL: {url}")
        return "\n".join(parts)

    def _labels(self, article: dict[str, Any]) -> list[str]:
        return [key for key, _title in self._label_entries(article)]

    def _label_entries(self, article: dict[str, Any]) -> list[tuple[str, str]]:
        raw = article.get("labels") or article.get("tags") or article.get("label")
        if isinstance(raw, dict):
            raw_values = list(raw.values())
        elif isinstance(raw, list):
            raw_values = raw
        elif isinstance(raw, str):
            raw_values = re.split(r"[,;|]", raw)
        else:
            raw_values = []

        labels: list[tuple[str, str]] = []
        seen: set[str] = set()
        for value in raw_values:
            if isinstance(value, dict):
                value = (
                    value.get("name")
                    or value.get("label")
                    or value.get("title")
                    or value.get("id")
                )
            title = re.sub(r"\s+", " ", str(value or "").strip().removeprefix("#")).strip()
            normalized = (
                re.sub(r"\s+", " ", str(value or "").strip().removeprefix("#"))
                .strip()
                .lower()
            )
            if normalized and normalized not in seen:
                seen.add(normalized)
                labels.append((normalized, title or normalized))
        return labels

    def _merge_label_entries(self, *groups: list[tuple[str, str]]) -> list[tuple[str, str]]:
        merged: dict[str, str] = {}
        for group in groups:
            for key, title in group:
                merged.setdefault(key, title)
        return [(key, merged[key]) for key in sorted(merged)]

    def _add_label_source_link(
        self,
        label_links: dict[str, dict[str, Any]],
        labels: list[tuple[str, str]],
        source_id: str,
        entity_type: str,
    ) -> None:
        for key, title in labels:
            info = label_links.setdefault(
                key,
                {
                    "title": title,
                    "article_source_ids": set(),
                    "highlight_source_ids": set(),
                },
            )
            if entity_type == "article":
                info["article_source_ids"].add(source_id)
            elif entity_type == "highlight":
                info["highlight_source_ids"].add(source_id)

    def _label_units(self, label_links: dict[str, dict[str, Any]], now: datetime) -> list[KnowledgeUnit]:
        units: list[KnowledgeUnit] = []
        for key in sorted(label_links):
            info = label_links[key]
            article_source_ids = sorted(info["article_source_ids"])
            highlight_source_ids = sorted(info["highlight_source_ids"])
            title = str(info["title"])
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.OMNIVORE_JSON,
                    source_id=self._label_source_id(key),
                    source_entity_type="label",
                    title=title,
                    content=f"Omnivore label: {title}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "label": title,
                        "normalized_label": key,
                        "article_count": len(article_source_ids),
                        "highlight_count": len(highlight_source_ids),
                        "article_source_ids": article_source_ids,
                        "highlight_source_ids": highlight_source_ids,
                        "linked_source_ids": sorted(article_source_ids + highlight_source_ids),
                    },
                    tags=["omnivore", "label", key],
                    created_at=now,
                    updated_at=now,
                )
            )
        return units

    def _label_edges(
        self,
        label_units: list[KnowledgeUnit],
        label_links: dict[str, dict[str, Any]],
        requested_content_types: set[str],
    ) -> list[KnowledgeEdge]:
        label_unit_ids = {unit.metadata["normalized_label"]: unit.source_id for unit in label_units}
        edges: list[KnowledgeEdge] = []
        for key, info in label_links.items():
            label_id = label_unit_ids.get(key)
            if not label_id:
                continue
            linked: list[tuple[str, str]] = []
            if "article" in requested_content_types:
                linked.extend(("article", source_id) for source_id in info["article_source_ids"])
            if "highlight" in requested_content_types:
                linked.extend(("highlight", source_id) for source_id in info["highlight_source_ids"])
            for entity_type, source_id in linked:
                edges.append(
                    KnowledgeEdge(
                        id=self._label_edge_id(source_id, label_id),
                        from_unit_id=source_id,
                        to_unit_id=label_id,
                        relation=EdgeRelation.RELATES_TO,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.OMNIVORE_JSON.value,
                            "from_entity_type": entity_type,
                            "to_entity_type": "label",
                            "relation_type": "content_label",
                            "label": info["title"],
                            "normalized_label": key,
                        },
                    )
                )
        return list({edge.id: edge for edge in edges}.values())

    def _label_source_id(self, label: str) -> str:
        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:24]
        return f"omnivore_label:{digest}"

    def _label_edge_id(self, content_source_id: str, label_source_id: str) -> str:
        digest = hashlib.sha256(f"{content_source_id}\0label\0{label_source_id}".encode("utf-8")).hexdigest()[:24]
        return f"omnivore_label_edge:{digest}"

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        normalized = {self._normalize_key(key): value for key, value in item.items()}
        for key in keys:
            value = normalized.get(self._normalize_key(key))
            if value is None or isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _is_archived(self, article: dict[str, Any], state: str) -> bool:
        archived = self._first(article, "archived", "isArchived", "is_archived")
        if archived:
            return self._is_truthy(archived)
        return state.strip().lower() in {"archived", "archive"}

    def _is_read(self, article: dict[str, Any], state: str, read_at: str) -> bool:
        read = self._first(article, "read", "isRead", "is_read")
        if read:
            return self._is_truthy(read)
        return bool(read_at) or state.strip().lower() in {"read", "completed"}

    def _is_truthy(self, value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "read", "archived"}

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        cleaned = value.strip()
        if re.fullmatch(r"\d+(?:\.0+)?", cleaned):
            try:
                return datetime.fromtimestamp(int(float(cleaned)), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        try:
            parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
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
