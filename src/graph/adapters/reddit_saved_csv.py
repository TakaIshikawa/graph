"""Adapter for Reddit saved items CSV exports."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class RedditSavedCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "reddit_saved_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["post", "comment"]

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
        post_index: dict[str, str] = {}
        comments: list[KnowledgeUnit] = []

        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None or unit.source_entity_type not in requested:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
                if unit.source_entity_type == "post":
                    self._index_post(post_index, unit)
                else:
                    comments.append(unit)

        included_source_ids = {unit.source_id for unit in result.units}
        emitted_edges: set[tuple[str, str]] = set()
        for comment in sorted(comments, key=lambda unit: unit.source_id):
            if comment.source_id not in included_source_ids:
                continue
            target_id = self._comment_target(post_index, comment)
            if not target_id or target_id not in included_source_ids:
                continue
            edge_key = (comment.source_id, target_id)
            if edge_key in emitted_edges:
                continue
            emitted_edges.add(edge_key)
            result.edges.append(
                KnowledgeEdge(
                    id=self._edge_id(comment.source_id, target_id),
                    from_unit_id=comment.source_id,
                    to_unit_id=target_id,
                    relation=EdgeRelation.REPLIES_TO,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.REDDIT_SAVED_CSV.value,
                        "from_entity_type": "comment",
                        "to_entity_type": "post",
                        "link_id": comment.metadata.get("link_id", ""),
                        "parent_id": comment.metadata.get("parent_id", ""),
                    },
                )
            )

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id, edge.id))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.csv") if child.is_file())

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            return [{str(k).strip(): v for k, v in row.items() if k is not None} for row in reader]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        entity_type = self._entity_type(row, source_file)
        title = self._first(row, "title", "link_title")
        body = self._first(row, "body", "selftext", "text")
        permalink = self._absolute_permalink(self._first(row, "permalink", "link_permalink"))
        url = self._first(row, "url", "link_url")
        if not title and not body and not permalink and not url:
            return None

        created_text = self._first(row, "created_utc", "created", "created_at", "saved_at")
        created_at = self._parse_datetime(created_text)
        now = datetime.now(timezone.utc)
        name = self._first(row, "name", "fullname")
        reddit_id = self._first(row, "id")
        link_id = self._normalize_fullname(self._first(row, "link_id"))
        parent_id = self._normalize_fullname(self._first(row, "parent_id"))
        metadata = {
            "id": reddit_id,
            "name": name,
            "fullname": name,
            "title": title,
            "body": body,
            "selftext": self._first(row, "selftext"),
            "subreddit": self._first(row, "subreddit", "subreddit_name_prefixed"),
            "author": self._first(row, "author"),
            "permalink": permalink,
            "url": url,
            "created_utc": created_at.isoformat() if created_at else created_text,
            "score": self._parse_int(self._first(row, "score")),
            "link_id": link_id,
            "parent_id": parent_id,
            "link_title": self._first(row, "link_title"),
            "source_file": source_file,
        }
        unit_title = title or (f"Comment on {metadata['link_title']}" if metadata["link_title"] else "Reddit saved comment")
        return KnowledgeUnit(
            source_project=SourceProject.REDDIT_SAVED_CSV,
            source_id=self._source_id(row, entity_type, permalink or url or title or body),
            source_entity_type=entity_type,
            title=unit_title,
            content=self._content(entity_type, title, body, metadata["link_title"], permalink, url),
            content_type=ContentType.ARTIFACT if entity_type == "post" else ContentType.INSIGHT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None)},
            tags=self._tags(entity_type, metadata["subreddit"]),
            created_at=created_at or now,
            updated_at=created_at or now,
        )

    def _entity_type(self, row: dict[str, Any], source_file: str) -> str:
        explicit = self._first(row, "type", "kind", "item_type").lower()
        name = self._first(row, "name", "fullname")
        if explicit in {"comment", "comments", "t1"} or name.startswith("t1_") or source_file.lower().endswith("comments.csv"):
            return "comment"
        if explicit in {"post", "submission", "link", "t3"} or name.startswith("t3_"):
            return "post"
        if self._first(row, "body") and not self._first(row, "title"):
            return "comment"
        return "post"

    def _source_id(self, row: dict[str, Any], entity_type: str, fallback: str) -> str:
        explicit = self._first(row, "name", "fullname") or self._first(row, "id") or self._absolute_permalink(
            self._first(row, "permalink", "link_permalink")
        )
        raw = explicit or "|".join([entity_type, fallback, self._first(row, "created_utc", "created", "created_at")])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"reddit_saved_csv:{digest}"

    def _index_post(self, index: dict[str, str], unit: KnowledgeUnit) -> None:
        for value in (
            self._normalize_fullname(self._string(unit.metadata.get("name"))),
            self._normalize_fullname(self._string(unit.metadata.get("id"))),
            self._post_id_from_permalink(self._string(unit.metadata.get("permalink"))),
        ):
            if value:
                index[value] = unit.source_id

    def _comment_target(self, post_index: dict[str, str], comment: KnowledgeUnit) -> str:
        for value in (
            self._normalize_fullname(self._string(comment.metadata.get("link_id"))),
            self._post_id_from_permalink(self._string(comment.metadata.get("permalink"))),
        ):
            if value and value in post_index:
                return post_index[value]
        return ""

    def _edge_id(self, from_id: str, to_id: str) -> str:
        raw = "|".join([SourceProject.REDDIT_SAVED_CSV.value, EdgeRelation.REPLIES_TO.value, from_id, to_id])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"reddit-saved-csv-replies-{digest}"

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

    def _tags(self, entity_type: str, subreddit: str) -> list[str]:
        tags = ["reddit", entity_type]
        if subreddit:
            tags.append(subreddit)
        return tags

    def _absolute_permalink(self, value: str) -> str:
        if not value:
            return ""
        if value.startswith("http://") or value.startswith("https://"):
            return value
        if value.startswith("/"):
            return f"https://www.reddit.com{value}"
        return value

    def _post_id_from_permalink(self, value: str) -> str:
        parts = [part for part in value.split("/") if part]
        if "comments" in parts:
            index = parts.index("comments")
            if index + 1 < len(parts):
                return f"t3_{parts[index + 1]}"
        return ""

    def _normalize_fullname(self, value: str) -> str:
        text = value.strip()
        if not text:
            return ""
        if text.startswith(("t1_", "t3_")):
            return text
        return f"t3_{text}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _string(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _parse_int(self, value: str) -> int | None:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, TypeError, ValueError):
            pass
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
