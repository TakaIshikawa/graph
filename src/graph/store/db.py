"""SQLite store for the knowledge graph."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from graph.store.migrations import SCHEMA_VERSION, ensure_schema
from graph.types.enums import EdgeSource
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState

if TYPE_CHECKING:
    from graph.adapters.base import IngestResult


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _row_to_unit(row: sqlite3.Row) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=row["id"],
        source_project=row["source_project"],
        source_id=row["source_id"],
        source_entity_type=row["source_entity_type"],
        title=row["title"],
        content=row["content"],
        content_type=row["content_type"],
        metadata=json.loads(row["metadata"]),
        tags=json.loads(row["tags"]),
        confidence=row["confidence"],
        utility_score=row["utility_score"],
        created_at=row["created_at"],
        ingested_at=row["ingested_at"],
        updated_at=row["updated_at"],
    )


def _row_to_edge(row: sqlite3.Row) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=row["id"],
        from_unit_id=row["from_unit_id"],
        to_unit_id=row["to_unit_id"],
        relation=row["relation"],
        weight=row["weight"],
        source=row["source"],
        metadata=json.loads(row["metadata"]),
        created_at=row["created_at"],
    )


def _row_to_saved_query(row: sqlite3.Row) -> dict:
    return {
        "name": row["name"],
        "query": row["query"],
        "mode": row["mode"],
        "limit": row["limit"],
        "filters": json.loads(row["filters"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


class Store:
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        ensure_schema(self.conn)

    def close(self) -> None:
        self.conn.close()

    # --- Unit CRUD ---

    def insert_unit(self, unit: KnowledgeUnit) -> KnowledgeUnit:
        if not unit.id:
            unit.id = _new_id()
        now = _utcnow_iso()
        self.conn.execute(
            """INSERT INTO knowledge_units
               (id, source_project, source_id, source_entity_type,
                title, content, content_type, metadata, tags,
                confidence, utility_score, embedding,
                created_at, ingested_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_project, source_id, source_entity_type)
               DO UPDATE SET
                   title = excluded.title,
                   content = excluded.content,
                   metadata = excluded.metadata,
                   tags = excluded.tags,
                   confidence = excluded.confidence,
                   utility_score = excluded.utility_score,
                   updated_at = excluded.updated_at
            """,
            (
                unit.id,
                unit.source_project,
                unit.source_id,
                unit.source_entity_type,
                unit.title,
                unit.content,
                unit.content_type,
                json.dumps(unit.metadata),
                json.dumps(unit.tags),
                unit.confidence,
                unit.utility_score,
                None,
                unit.created_at.isoformat()
                if isinstance(unit.created_at, datetime)
                else str(unit.created_at),
                now,
                now,
            ),
        )
        self.conn.commit()
        return unit

    def get_unit(self, unit_id: str) -> KnowledgeUnit | None:
        row = self.conn.execute(
            "SELECT * FROM knowledge_units WHERE id = ?", (unit_id,)
        ).fetchone()
        return _row_to_unit(row) if row else None

    def get_unit_by_source(
        self, source_project: str, source_id: str, source_entity_type: str
    ) -> KnowledgeUnit | None:
        row = self.conn.execute(
            """SELECT * FROM knowledge_units
               WHERE source_project = ? AND source_id = ? AND source_entity_type = ?""",
            (source_project, source_id, source_entity_type),
        ).fetchone()
        return _row_to_unit(row) if row else None

    def get_all_units(self, *, limit: int = 10000) -> list[KnowledgeUnit]:
        rows = self.conn.execute(
            "SELECT * FROM knowledge_units ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_unit(r) for r in rows]

    def get_units_with_embeddings(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
    ) -> list[tuple[KnowledgeUnit, bytes]]:
        query = "SELECT * FROM knowledge_units WHERE embedding IS NOT NULL"
        params: list = []
        if source_project:
            query += " AND source_project = ?"
            params.append(source_project)
        if content_type:
            query += " AND content_type = ?"
            params.append(content_type)
        rows = self.conn.execute(query, params).fetchall()
        return [(_row_to_unit(r), r["embedding"]) for r in rows]

    def update_embedding(self, unit_id: str, embedding: bytes) -> None:
        self.conn.execute(
            "UPDATE knowledge_units SET embedding = ?, updated_at = ? WHERE id = ?",
            (embedding, _utcnow_iso(), unit_id),
        )
        self.conn.commit()

    # --- JSON import/export ---

    def export_json(self) -> dict:
        """Return a portable JSON-serializable graph backup."""
        units = [
            {
                "id": unit.id,
                "source_project": str(unit.source_project),
                "source_id": unit.source_id,
                "source_entity_type": unit.source_entity_type,
                "title": unit.title,
                "content": unit.content,
                "content_type": str(unit.content_type),
                "metadata": unit.metadata,
                "tags": unit.tags,
                "confidence": unit.confidence,
                "utility_score": unit.utility_score,
                "created_at": _json_value(unit.created_at),
                "ingested_at": _json_value(unit.ingested_at),
                "updated_at": _json_value(unit.updated_at),
            }
            for unit in self.get_all_units(limit=1000000000)
        ]
        edges = [
            {
                "id": edge.id,
                "from_unit_id": edge.from_unit_id,
                "to_unit_id": edge.to_unit_id,
                "relation": str(edge.relation),
                "weight": edge.weight,
                "source": str(edge.source),
                "metadata": edge.metadata,
                "created_at": _json_value(edge.created_at),
            }
            for edge in self.get_all_edges()
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "exported_at": _utcnow_iso(),
            "units": units,
            "edges": edges,
        }

    def import_json(self, payload: dict) -> dict:
        """Import a portable graph backup idempotently.

        Unit IDs are preserved for new rows. If a unit already exists by source
        identity, that database ID is kept and imported edges are remapped to it.
        """
        schema_version = payload.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported graph JSON schema_version {schema_version!r}; "
                f"expected {SCHEMA_VERSION}"
            )

        units_inserted = 0
        units_updated = 0
        edges_inserted = 0
        edges_skipped = 0
        imported_to_graph_id: dict[str, str] = {}

        for data in payload.get("units", []):
            unit = KnowledgeUnit(**data)
            existing = self.get_unit_by_source(
                str(unit.source_project), unit.source_id, unit.source_entity_type
            )
            if existing:
                unit.id = existing.id
                units_updated += 1
            else:
                units_inserted += 1

            saved = self.insert_unit(unit)
            actual_id = unit.id or saved.id
            imported_to_graph_id[data["id"]] = actual_id
            fetched = self.get_unit(actual_id)
            if fetched:
                self.fts_index_unit(fetched)

        for data in payload.get("edges", []):
            from_id = imported_to_graph_id.get(data["from_unit_id"], data["from_unit_id"])
            to_id = imported_to_graph_id.get(data["to_unit_id"], data["to_unit_id"])
            before = self.conn.total_changes
            edge = KnowledgeEdge(
                id=data.get("id", ""),
                from_unit_id=from_id,
                to_unit_id=to_id,
                relation=data["relation"],
                weight=data.get("weight", 1.0),
                source=data.get("source", EdgeSource.INFERRED),
                metadata=data.get("metadata", {}),
                created_at=data.get("created_at") or _utcnow_iso(),
            )
            self.insert_edge(edge)
            if self.conn.total_changes > before:
                edges_inserted += 1
            else:
                edges_skipped += 1

        return {
            "units_inserted": units_inserted,
            "units_updated": units_updated,
            "edges_inserted": edges_inserted,
            "edges_skipped": edges_skipped,
        }

    def count_units(self, *, source_project: str | None = None) -> int:
        if source_project:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM knowledge_units WHERE source_project = ?",
                (source_project,),
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM knowledge_units").fetchone()
        return row[0]

    # --- Saved queries ---

    def save_query(
        self,
        *,
        name: str,
        query: str,
        mode: str = "fulltext",
        limit: int = 10,
        filters: dict | None = None,
    ) -> dict:
        now = _utcnow_iso()
        normalized_filters = filters or {}
        self.conn.execute(
            """INSERT INTO saved_queries
               (name, query, mode, "limit", filters, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(name)
               DO UPDATE SET
                   query = excluded.query,
                   mode = excluded.mode,
                   "limit" = excluded."limit",
                   filters = excluded.filters,
                   updated_at = excluded.updated_at
            """,
            (
                name,
                query,
                mode,
                limit,
                json.dumps(normalized_filters, sort_keys=True),
                now,
                now,
            ),
        )
        self.conn.commit()
        saved = self.get_saved_query(name)
        if saved is None:
            raise RuntimeError(f"Saved query was not written: {name}")
        return saved

    def get_saved_query(self, name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM saved_queries WHERE name = ?", (name,)
        ).fetchone()
        return _row_to_saved_query(row) if row else None

    def list_saved_queries(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM saved_queries ORDER BY name"
        ).fetchall()
        return [_row_to_saved_query(row) for row in rows]

    def delete_saved_query(self, name: str) -> bool:
        cursor = self.conn.execute(
            "DELETE FROM saved_queries WHERE name = ?", (name,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # --- Edge CRUD ---

    def insert_edge(self, edge: KnowledgeEdge) -> KnowledgeEdge:
        if not edge.id:
            edge.id = _new_id()
        self.conn.execute(
            """INSERT INTO edges
               (id, from_unit_id, to_unit_id, relation, weight, source, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(from_unit_id, to_unit_id, relation) DO NOTHING
            """,
            (
                edge.id,
                edge.from_unit_id,
                edge.to_unit_id,
                edge.relation,
                edge.weight,
                edge.source,
                json.dumps(edge.metadata),
                edge.created_at.isoformat()
                if isinstance(edge.created_at, datetime)
                else str(edge.created_at),
            ),
        )
        self.conn.commit()
        return edge

    def get_all_edges(self) -> list[KnowledgeEdge]:
        rows = self.conn.execute("SELECT * FROM edges").fetchall()
        return [_row_to_edge(r) for r in rows]

    def get_edges_for_unit(self, unit_id: str) -> list[KnowledgeEdge]:
        rows = self.conn.execute(
            "SELECT * FROM edges WHERE from_unit_id = ? OR to_unit_id = ?",
            (unit_id, unit_id),
        ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def edge_exists_between(self, left_unit_id: str, right_unit_id: str) -> bool:
        """Return true if any direct edge exists between two units in either direction."""
        row = self.conn.execute(
            """SELECT 1 FROM edges
               WHERE (from_unit_id = ? AND to_unit_id = ?)
                  OR (from_unit_id = ? AND to_unit_id = ?)
               LIMIT 1""",
            (left_unit_id, right_unit_id, right_unit_id, left_unit_id),
        ).fetchone()
        return row is not None

    # --- Sync state ---

    def get_sync_state(
        self, source_project: str, source_entity_type: str
    ) -> SyncState | None:
        row = self.conn.execute(
            """SELECT * FROM sync_state
               WHERE source_project = ? AND source_entity_type = ?""",
            (source_project, source_entity_type),
        ).fetchone()
        if not row:
            return None
        return SyncState(
            source_project=row["source_project"],
            source_entity_type=row["source_entity_type"],
            last_sync_at=row["last_sync_at"],
            last_source_id=row["last_source_id"],
            items_synced=row["items_synced"],
        )

    def upsert_sync_state(self, state: SyncState) -> None:
        self.conn.execute(
            """INSERT INTO sync_state
               (source_project, source_entity_type, last_sync_at, last_source_id, items_synced)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(source_project, source_entity_type)
               DO UPDATE SET
                   last_sync_at = excluded.last_sync_at,
                   last_source_id = excluded.last_source_id,
                   items_synced = sync_state.items_synced + excluded.items_synced
            """,
            (
                state.source_project,
                state.source_entity_type,
                state.last_sync_at.isoformat()
                if isinstance(state.last_sync_at, datetime)
                else str(state.last_sync_at),
                state.last_source_id,
                state.items_synced,
            ),
        )
        self.conn.commit()

    # --- FTS ---

    def fts_index_unit(self, unit: KnowledgeUnit) -> None:
        self.conn.execute(
            "DELETE FROM knowledge_fts WHERE unit_id = ?", (unit.id,)
        )
        self.conn.execute(
            "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
            (unit.id, unit.title, unit.content, " ".join(unit.tags)),
        )
        self.conn.commit()

    def fts_search(self, query: str, *, limit: int = 20) -> list[dict]:
        try:
            rows = self.conn.execute(
                """SELECT unit_id, rank
                   FROM knowledge_fts
                   WHERE knowledge_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit),
            ).fetchall()
            return [{"unit_id": r["unit_id"], "rank": r["rank"]} for r in rows]
        except sqlite3.OperationalError:
            # Fallback to LIKE search if FTS query syntax is invalid
            pattern = f"%{query}%"
            rows = self.conn.execute(
                """SELECT id as unit_id, -1.0 as rank
                   FROM knowledge_units
                   WHERE title LIKE ? OR content LIKE ?
                   LIMIT ?""",
                (pattern, pattern, limit),
            ).fetchall()
            return [{"unit_id": r["unit_id"], "rank": r["rank"]} for r in rows]

    # --- Ingestion orchestration ---

    def ingest(self, result: "IngestResult", source_project: str) -> dict:
        """Ingest adapter results: insert units, remap edge IDs, insert edges.

        Returns stats dict with units_inserted, units_skipped, edges_inserted.
        """
        units_inserted = 0
        units_skipped = 0

        # Build mapping from source_id -> graph unit id for edge remapping
        source_to_graph_id: dict[str, str] = {}

        for unit in result.units:
            existing = self.get_unit_by_source(
                unit.source_project, unit.source_id, unit.source_entity_type
            )
            if existing:
                # Update existing unit
                unit.id = existing.id
                self.insert_unit(unit)  # UPSERT
                self.fts_index_unit(unit)
                source_to_graph_id[unit.source_id] = existing.id
                units_skipped += 1
            else:
                inserted = self.insert_unit(unit)
                source_to_graph_id[unit.source_id] = inserted.id
                self.fts_index_unit(inserted)
                units_inserted += 1

        # Remap and insert edges
        edges_inserted = 0
        for edge in result.edges:
            edge_source_project = edge.metadata.get("source_project", source_project)
            # Resolve source-local IDs to graph IDs
            from_id = source_to_graph_id.get(edge.from_unit_id)
            to_id = source_to_graph_id.get(edge.to_unit_id)

            if not from_id:
                # Try finding in existing graph data
                from_unit = self.get_unit_by_source(
                    edge_source_project, edge.from_unit_id, self._guess_entity_type(edge, "from")
                )
                from_id = from_unit.id if from_unit else None

            if not to_id:
                to_unit = self.get_unit_by_source(
                    edge_source_project, edge.to_unit_id, self._guess_entity_type(edge, "to")
                )
                to_id = to_unit.id if to_unit else None

            if from_id and to_id:
                edge.from_unit_id = from_id
                edge.to_unit_id = to_id
                self.insert_edge(edge)
                edges_inserted += 1

        return {
            "units_inserted": units_inserted,
            "units_skipped": units_skipped,
            "edges_inserted": edges_inserted,
        }

    def _guess_entity_type(self, edge: KnowledgeEdge, direction: str) -> str:
        """Guess entity type for edge resolution based on source metadata."""
        explicit_type = edge.metadata.get(f"{direction}_entity_type")
        if explicit_type:
            return explicit_type
        source_project = edge.metadata.get("source_project", "")
        if source_project == "max":
            return "insight" if direction == "from" else "buildable_unit"
        if source_project == "forty_two" or edge.source == EdgeSource.SOURCE:
            return "knowledge_node"
        return "knowledge_node"
