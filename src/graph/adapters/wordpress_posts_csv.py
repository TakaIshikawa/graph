"""Adapter for WordPress posts CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class WordPressPostsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "wordpress_posts_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["post"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "post" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit is None or (
                    sync_at
                    and (
                        not any(key in unit.metadata for key in ["published_at", "modified_at"])
                        or unit.updated_at <= sync_at
                    )
                ):
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        post_id = first(row, "ID", "Post ID", "post_id")
        title = first(row, "Title", "Post Title")
        slug = first(row, "Slug", "Post Slug")
        status = first(row, "Status", "Post Status")
        author = first(row, "Author", "Post Author")
        date_text = first(row, "Date", "Post Date", "Published At", "Created At")
        modified_text = first(row, "Modified", "Modified Date", "Updated At")
        url = first(row, "URL", "Permalink", "Link")
        categories_raw = first(row, "Categories", "Category")
        tags_raw = first(row, "Tags", "Post Tags")
        excerpt = first(row, "Excerpt", "Summary")
        body = first(row, "Content", "Post Content", "Body")
        published_at = parse_datetime(date_text)
        modified_at = parse_datetime(modified_text)
        if not any([post_id, title, slug, status, author, date_text, modified_text, url, categories_raw, tags_raw, excerpt, body]):
            return None
        now = datetime.now(timezone.utc)
        created = published_at or modified_at or now
        updated = modified_at or published_at or now
        categories = split_values(categories_raw)
        post_tags = split_values(tags_raw)
        metadata = clean_metadata(
            {
                "post_id": post_id,
                "slug": slug,
                "status": status,
                "author": author,
                "published_at": published_at.isoformat() if published_at else date_text,
                "modified_at": modified_at.isoformat() if modified_at else modified_text,
                "url": url,
                "categories": categories,
                "tags": post_tags,
                "categories_raw": categories_raw,
                "tags_raw": tags_raw,
                "excerpt": excerpt,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        content = body or excerpt or self._content(metadata)
        return KnowledgeUnit(
            source_project="wordpress_posts_csv",
            source_id=f"wordpress_posts_csv:{post_id}" if post_id else digest_source_id("wordpress_posts_csv", title, slug, url, date_text, index),
            source_entity_type="post",
            title=title or slug or url or "WordPress post",
            content=content,
            content_type=ContentType.ARTIFACT if body or excerpt else ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["wordpress", "post", status, author, *categories, *post_tags] if tag)),
            created_at=created,
            updated_at=updated,
        )

    def _content(self, metadata: dict[str, Any]) -> str:
        labels = [("Slug", "slug"), ("Status", "status"), ("Author", "author"), ("URL", "url")]
        return "\n".join(f"{label}: {metadata[key]}" for label, key in labels if key in metadata)
