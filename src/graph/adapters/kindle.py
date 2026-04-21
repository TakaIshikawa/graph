"""Adapter for Kindle highlights from supabooks database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class KindleAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "kindle"

    @property
    def entity_types(self) -> list[str]:
        return ["book", "highlight"]

    def __init__(self, db_path: str = "") -> None:
        self.db_path = db_path

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

        # Map to track book source_id -> graph_id for edge creation
        book_id_map = {}

        if "book" in types:
            book_id_map = self._ingest_books(conn, result, since)

        if "highlight" in types:
            self._ingest_highlights(conn, result, since, book_id_map)

        conn.close()
        return result

    def _ingest_books(
        self, conn: sqlite3.Connection, result: IngestResult, since: SyncState | None
    ) -> dict[int, str]:
        """Ingest books as METADATA nodes. Returns map of book.id -> source_id."""
        where = ""
        params: list = []
        if since and since.last_sync_at:
            where = "WHERE created_at > ?"
            params.append(
                since.last_sync_at.isoformat()
                if hasattr(since.last_sync_at, "isoformat")
                else str(since.last_sync_at)
            )

        rows = conn.execute(
            f"""SELECT id, title, author, metadata, created_at
                FROM books {where}
                ORDER BY created_at""",
            params,
        ).fetchall()

        book_id_map = {}

        for row in rows:
            source_id = f"book_{row['id']}"
            book_id_map[row["id"]] = source_id

            # Get highlight count for this book
            highlight_count = conn.execute(
                "SELECT COUNT(*) as count FROM highlights WHERE book_id = ?",
                (row["id"],)
            ).fetchone()["count"]

            unit = KnowledgeUnit(
                source_project=SourceProject.KINDLE,
                source_id=source_id,
                source_entity_type="book",
                title=f"{row['title']} by {row['author']}",
                content=f"Book: {row['title']}\nAuthor: {row['author']}\nHighlights: {highlight_count}",
                content_type=ContentType.METADATA,
                metadata={
                    "book_title": row["title"],
                    "author": row["author"],
                    "highlight_count": highlight_count,
                },
                tags=[row["author"].split(",")[0].strip()],  # Last name as tag
                created_at=row["created_at"] or "1970-01-01T00:00:00+00:00",
            )
            result.units.append(unit)

        return book_id_map

    def _ingest_highlights(
        self,
        conn: sqlite3.Connection,
        result: IngestResult,
        since: SyncState | None,
        book_id_map: dict[int, str],
    ) -> None:
        """Ingest highlights as INSIGHT nodes."""
        where = ""
        params: list = []
        if since and since.last_sync_at:
            where = "WHERE h.created_at > ?"
            params.append(
                since.last_sync_at.isoformat()
                if hasattr(since.last_sync_at, "isoformat")
                else str(since.last_sync_at)
            )

        rows = conn.execute(
            f"""SELECT h.id, h.book_id, h.content, h.highlight_type,
                       h.location, h.page, h.date_added, h.created_at,
                       b.title as book_title, b.author as book_author
                FROM highlights h
                JOIN books b ON h.book_id = b.id
                {where}
                ORDER BY h.created_at""",
            params,
        ).fetchall()

        for row in rows:
            source_id = f"highlight_{row['id']}"

            # Construct title from book + location/page
            location_str = f"Location {row['location']}" if row['location'] else ""
            page_str = f"Page {row['page']}" if row['page'] else ""
            position = " - ".join(filter(None, [location_str, page_str]))
            title = f"{row['book_title']}: {position}" if position else row['book_title']

            unit = KnowledgeUnit(
                source_project=SourceProject.KINDLE,
                source_id=source_id,
                source_entity_type="highlight",
                title=title,
                content=row["content"],
                content_type=ContentType.INSIGHT,
                metadata={
                    "book_id": row["book_id"],
                    "book_title": row["book_title"],
                    "author": row["book_author"],
                    "highlight_type": row["highlight_type"],
                    "location": row["location"],
                    "page": row["page"],
                    "date_added": row["date_added"],
                },
                tags=[row["book_author"].split(",")[0].strip()],  # Author last name
                created_at=row["created_at"] or "1970-01-01T00:00:00+00:00",
            )
            result.units.append(unit)

            # Create edge: book -> contains -> highlight
            # Note: Store.ingest() will handle ID remapping from source IDs to graph IDs
            book_source_id = book_id_map.get(row["book_id"]) or f"book_{row['book_id']}"

            edge = KnowledgeEdge(
                from_unit_id=book_source_id,
                to_unit_id=source_id,
                relation=EdgeRelation.RELATES_TO,  # Using RELATES_TO for "contains" relationship
                weight=1.0,
                source=EdgeSource.SOURCE,
                metadata={"relation_type": "contains"},
            )
            result.edges.append(edge)
