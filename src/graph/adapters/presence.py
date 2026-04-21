"""Adapter for the presence project (content generation + knowledge base)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class PresenceAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "presence"

    @property
    def entity_types(self) -> list[str]:
        return ["knowledge_item", "generated_content"]

    def __init__(self, db_path: str = "", min_score: float = 7.0) -> None:
        self.db_path = db_path
        self.min_score = min_score

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if not Path(self.db_path).exists():
            return result

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        types = entity_types or self.entity_types

        if "knowledge_item" in types:
            self._ingest_knowledge(conn, result, since)

        if "generated_content" in types:
            self._ingest_content(conn, result, since)

        conn.close()
        return result

    def _ingest_knowledge(
        self, conn: sqlite3.Connection, result: IngestResult, since: SyncState | None
    ) -> None:
        where = "WHERE approved = 1"
        params: list = []
        if since and since.last_sync_at:
            where += " AND created_at > ?"
            params.append(
                since.last_sync_at.isoformat()
                if hasattr(since.last_sync_at, "isoformat")
                else str(since.last_sync_at)
            )

        rows = conn.execute(
            f"""SELECT id, source_type, source_id, source_url, author,
                       content, insight, created_at
                FROM knowledge {where}
                ORDER BY created_at""",
            params,
        ).fetchall()

        for row in rows:
            main_content = row["insight"] or row["content"]
            title = main_content[:80] if main_content else "Untitled"

            unit = KnowledgeUnit(
                source_project=SourceProject.PRESENCE,
                source_id=str(row["id"]),
                source_entity_type="knowledge_item",
                title=title,
                content=main_content or "",
                content_type=ContentType.INSIGHT,
                metadata={
                    "source_type": row["source_type"],
                    "original_source_id": row["source_id"],
                    "source_url": row["source_url"],
                    "author": row["author"],
                },
                tags=[row["source_type"]] if row["source_type"] else [],
                created_at=row["created_at"] or datetime.now(timezone.utc).isoformat(),
            )
            result.units.append(unit)

    def _ingest_content(
        self, conn: sqlite3.Connection, result: IngestResult, since: SyncState | None
    ) -> None:
        where = "WHERE published = 1 AND eval_score >= ?"
        params: list = [self.min_score]
        if since and since.last_sync_at:
            where += " AND created_at > ?"
            params.append(
                since.last_sync_at.isoformat()
                if hasattr(since.last_sync_at, "isoformat")
                else str(since.last_sync_at)
            )

        rows = conn.execute(
            f"""SELECT id, content_type, content, eval_score,
                       published_url, created_at
                FROM generated_content {where}
                ORDER BY created_at""",
            params,
        ).fetchall()

        for row in rows:
            content = row["content"] or ""
            title = content[:80] if content else "Untitled"

            unit = KnowledgeUnit(
                source_project=SourceProject.PRESENCE,
                source_id=f"gc-{row['id']}",
                source_entity_type="generated_content",
                title=title,
                content=content,
                content_type=ContentType.ARTIFACT,
                metadata={
                    "content_type": row["content_type"],
                    "eval_score": row["eval_score"],
                    "published_url": row["published_url"],
                },
                tags=[row["content_type"]] if row["content_type"] else [],
                utility_score=row["eval_score"] / 10.0 if row["eval_score"] else None,
                created_at=row["created_at"] or datetime.now(timezone.utc).isoformat(),
            )
            result.units.append(unit)
