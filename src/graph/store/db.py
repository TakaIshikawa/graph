"""SQLite store for the knowledge graph."""

from __future__ import annotations

import json
import re
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


def _excerpt(text: str, query: str, *, length: int = 160) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""

    terms = [
        term.lower()
        for term in re.findall(r"[\w-]+", query)
        if term.upper() not in {"AND", "OR", "NOT", "NEAR"}
    ]
    lower_text = text.lower()
    positions = [lower_text.find(term) for term in terms if lower_text.find(term) >= 0]
    if positions:
        start = max(min(positions) - length // 3, 0)
    else:
        start = 0
    snippet = text[start : start + length].strip()
    if start > 0:
        snippet = "..." + snippet
    if start + length < len(text):
        snippet += "..."
    return snippet


def _fallback_search_terms(query: str) -> list[str]:
    terms = [
        term
        for term in re.findall(r"[\w-]+", query)
        if term.upper() not in {"AND", "OR", "NOT", "NEAR"}
    ]
    return terms or [query]


def _requires_exact_single_term_filter(query: str) -> bool:
    stripped = query.strip()
    return bool(stripped) and not re.search(r"\s", stripped) and bool(
        re.search(r"[-_/]", stripped)
    )


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
                confidence, utility_score, embedding, embedding_updated_at,
                created_at, ingested_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                None,
                unit.created_at.isoformat()
                if isinstance(unit.created_at, datetime)
                else str(unit.created_at),
                now,
                unit.updated_at.isoformat()
                if isinstance(unit.updated_at, datetime)
                else str(unit.updated_at),
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
            "UPDATE knowledge_units SET embedding = ?, embedding_updated_at = ? WHERE id = ?",
            (embedding, _utcnow_iso(), unit_id),
        )
        self.conn.commit()

    def get_embedding_status(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, int]:
        where, params = self._unit_filter_sql(
            source_project=source_project,
            content_type=content_type,
        )
        row = self.conn.execute(
            f"""SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN embedding IS NULL THEN 1 ELSE 0 END) AS missing,
                    SUM(CASE
                        WHEN embedding IS NOT NULL
                         AND embedding_updated_at IS NOT NULL
                         AND embedding_updated_at >= updated_at
                        THEN 1 ELSE 0 END) AS fresh,
                    SUM(CASE
                        WHEN embedding IS NOT NULL
                         AND (embedding_updated_at IS NULL OR updated_at > embedding_updated_at)
                        THEN 1 ELSE 0 END) AS stale
                FROM knowledge_units
                {where}""",
            params,
        ).fetchone()
        return {
            "total": row["total"] or 0,
            "missing": row["missing"] or 0,
            "fresh": row["fresh"] or 0,
            "stale": row["stale"] or 0,
        }

    def get_units_for_embedding_refresh(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
        force: bool = False,
        stale_only: bool = False,
        limit: int | None = None,
    ) -> list[KnowledgeUnit]:
        where_parts, params = self._unit_filter_parts(
            source_project=source_project,
            content_type=content_type,
        )
        if force:
            pass
        elif stale_only:
            where_parts.append(
                """(embedding IS NULL
                    OR embedding_updated_at IS NULL
                    OR updated_at > embedding_updated_at)"""
            )
        else:
            where_parts.append("embedding IS NULL")

        query = "SELECT * FROM knowledge_units"
        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, limit))

        rows = self.conn.execute(query, params).fetchall()
        return [_row_to_unit(r) for r in rows]

    def _unit_filter_parts(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
    ) -> tuple[list[str], list]:
        where_parts: list[str] = []
        params: list = []
        if source_project:
            where_parts.append("source_project = ?")
            params.append(source_project)
        if content_type:
            where_parts.append("content_type = ?")
            params.append(content_type)
        return where_parts, params

    def _unit_filter_sql(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
    ) -> tuple[str, list]:
        where_parts, params = self._unit_filter_parts(
            source_project=source_project,
            content_type=content_type,
        )
        where = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        return where, params

    def update_unit_fields(
        self,
        unit_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        content_type: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> KnowledgeUnit | None:
        unit = self.get_unit(unit_id)
        if unit is None:
            return None

        if title is not None:
            unit.title = title
        if content is not None:
            unit.content = content
        if content_type is not None:
            unit.content_type = content_type
        if tags:
            for tag in tags:
                if tag not in unit.tags:
                    unit.tags.append(tag)
        if metadata:
            unit.metadata = {**unit.metadata, **metadata}

        now = _utcnow_iso()
        self.conn.execute(
            """UPDATE knowledge_units
               SET title = ?,
                   content = ?,
                   content_type = ?,
                   metadata = ?,
                   tags = ?,
                   updated_at = ?
               WHERE id = ?""",
            (
                unit.title,
                unit.content,
                unit.content_type,
                json.dumps(unit.metadata),
                json.dumps(unit.tags),
                now,
                unit.id,
            ),
        )
        self.conn.commit()
        updated = self.get_unit(unit_id)
        if updated is not None:
            self.fts_index_unit(updated)
        return updated

    def pin_unit(self, unit_id: str, *, reason: str | None = None) -> KnowledgeUnit | None:
        unit = self.get_unit(unit_id)
        if unit is None:
            return None

        metadata = dict(unit.metadata)
        now = _utcnow_iso()
        metadata["pinned"] = True
        metadata["pinned_at"] = now
        if reason is not None:
            metadata["pin_reason"] = reason
        else:
            metadata.pop("pin_reason", None)

        self.conn.execute(
            """UPDATE knowledge_units
               SET metadata = ?, updated_at = ?
               WHERE id = ?""",
            (json.dumps(metadata), now, unit.id),
        )
        self.conn.commit()
        updated = self.get_unit(unit_id)
        if updated is not None:
            self.fts_index_unit(updated)
        return updated

    def unpin_unit(self, unit_id: str) -> KnowledgeUnit | None:
        unit = self.get_unit(unit_id)
        if unit is None:
            return None

        metadata = dict(unit.metadata)
        for key in ("pinned", "pinned_at", "pin_reason"):
            metadata.pop(key, None)

        now = _utcnow_iso()
        self.conn.execute(
            """UPDATE knowledge_units
               SET metadata = ?, updated_at = ?
               WHERE id = ?""",
            (json.dumps(metadata), now, unit.id),
        )
        self.conn.commit()
        updated = self.get_unit(unit_id)
        if updated is not None:
            self.fts_index_unit(updated)
        return updated

    def rename_tag(
        self,
        old_tag: str,
        new_tag: str,
        *,
        dry_run: bool = False,
        source_project: str | None = None,
        content_type: str | None = None,
    ) -> dict:
        old_tag = old_tag.strip()
        new_tag = new_tag.strip()
        if not old_tag:
            raise ValueError("old_tag must not be empty.")
        if not new_tag:
            raise ValueError("new_tag must not be empty.")

        where, params = self._unit_filter_sql(
            source_project=source_project,
            content_type=content_type,
        )
        rows = self.conn.execute(
            f"SELECT * FROM knowledge_units {where} ORDER BY created_at DESC",
            params,
        ).fetchall()
        units = [_row_to_unit(row) for row in rows]

        changed: list[tuple[KnowledgeUnit, list[str]]] = []
        for unit in units:
            if old_tag not in unit.tags:
                continue

            renamed_tags: list[str] = []
            for tag in unit.tags:
                candidate = new_tag if tag == old_tag else tag
                if candidate not in renamed_tags:
                    renamed_tags.append(candidate)

            if renamed_tags != unit.tags:
                changed.append((unit, renamed_tags))

        changed_units = [
            {
                "id": unit.id,
                "title": unit.title,
                "source_project": str(unit.source_project),
                "source_entity_type": unit.source_entity_type,
                "content_type": str(unit.content_type),
                "old_tags": unit.tags,
                "new_tags": renamed_tags,
            }
            for unit, renamed_tags in changed
        ]

        if not dry_run and changed:
            now = _utcnow_iso()
            with self.conn:
                for unit, renamed_tags in changed:
                    self.conn.execute(
                        """UPDATE knowledge_units
                           SET tags = ?, updated_at = ?
                           WHERE id = ?""",
                        (json.dumps(renamed_tags), now, unit.id),
                    )
                    unit.tags = renamed_tags
                    unit.updated_at = now
                    self.conn.execute(
                        "DELETE FROM knowledge_fts WHERE unit_id = ?", (unit.id,)
                    )
                    self.conn.execute(
                        "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
                        (unit.id, unit.title, unit.content, " ".join(unit.tags)),
                    )

        return {
            "old_tag": old_tag,
            "new_tag": new_tag,
            "dry_run": dry_run,
            "changed_count": len(changed_units),
            "changed_units": changed_units,
            "filters": {
                "source_project": source_project,
                "content_type": content_type,
            },
        }

    def apply_tags_to_units(
        self,
        unit_ids: list[str],
        *,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict:
        add_tags = list(dict.fromkeys(tag.strip() for tag in (add_tags or []) if tag.strip()))
        remove_tags = list(
            dict.fromkeys(tag.strip() for tag in (remove_tags or []) if tag.strip())
        )
        if not add_tags and not remove_tags:
            raise ValueError("At least one --add or --remove tag is required.")
        overlap = sorted(set(add_tags) & set(remove_tags))
        if overlap:
            raise ValueError(
                "Tags cannot be both added and removed: " + ", ".join(overlap)
            )

        ordered_ids = list(dict.fromkeys(unit_ids))
        units = [unit for unit_id in ordered_ids if (unit := self.get_unit(unit_id))]

        changed: list[tuple[KnowledgeUnit, list[str]]] = []
        remove_set = set(remove_tags)
        for unit in units:
            next_tags = [tag for tag in unit.tags if tag not in remove_set]
            for tag in add_tags:
                if tag not in next_tags:
                    next_tags.append(tag)
            if next_tags != unit.tags:
                changed.append((unit, next_tags))

        changed_units = [
            {
                "id": unit.id,
                "title": unit.title,
                "source_project": str(unit.source_project),
                "source_entity_type": unit.source_entity_type,
                "content_type": str(unit.content_type),
                "old_tags": unit.tags,
                "new_tags": next_tags,
            }
            for unit, next_tags in changed
        ]

        if not dry_run and changed:
            now = _utcnow_iso()
            with self.conn:
                for unit, next_tags in changed:
                    self.conn.execute(
                        """UPDATE knowledge_units
                           SET tags = ?, updated_at = ?
                           WHERE id = ?""",
                        (json.dumps(next_tags), now, unit.id),
                    )
                    unit.tags = next_tags
                    unit.updated_at = now
                    self.conn.execute(
                        "DELETE FROM knowledge_fts WHERE unit_id = ?", (unit.id,)
                    )
                    self.conn.execute(
                        "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
                        (unit.id, unit.title, unit.content, " ".join(unit.tags)),
                    )

        return {
            "add_tags": add_tags,
            "remove_tags": remove_tags,
            "dry_run": dry_run,
            "matched_count": len(units),
            "changed_count": len(changed_units),
            "affected_count": len(changed_units),
            "changed_units": changed_units,
            "affected_units": changed_units,
        }

    def delete_unit(self, unit_id: str) -> dict:
        unit = self.get_unit(unit_id)
        if unit is None:
            return {"unit_id": unit_id, "deleted": False, "edges_deleted": 0}

        edge_cursor = self.conn.execute(
            "DELETE FROM edges WHERE from_unit_id = ? OR to_unit_id = ?",
            (unit_id, unit_id),
        )
        self.conn.execute("DELETE FROM knowledge_fts WHERE unit_id = ?", (unit_id,))
        unit_cursor = self.conn.execute(
            "DELETE FROM knowledge_units WHERE id = ?", (unit_id,)
        )
        self.conn.commit()
        return {
            "unit_id": unit_id,
            "deleted": unit_cursor.rowcount > 0,
            "edges_deleted": edge_cursor.rowcount,
        }

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

    def get_edge(self, edge_id: str) -> KnowledgeEdge | None:
        row = self.conn.execute("SELECT * FROM edges WHERE id = ?", (edge_id,)).fetchone()
        return _row_to_edge(row) if row else None

    def get_edges_for_unit(self, unit_id: str) -> list[KnowledgeEdge]:
        rows = self.conn.execute(
            """SELECT * FROM edges
               WHERE from_unit_id = ? OR to_unit_id = ?
               ORDER BY created_at DESC, id""",
            (unit_id, unit_id),
        ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def update_edge_fields(
        self,
        edge_id: str,
        *,
        relation: str | None = None,
        weight: float | None = None,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> KnowledgeEdge | None:
        edge = self.get_edge(edge_id)
        if edge is None:
            return None

        if relation is not None:
            edge.relation = relation
        if weight is not None:
            edge.weight = weight
        if source is not None:
            edge.source = source
        if metadata:
            edge.metadata = {**edge.metadata, **metadata}

        self.conn.execute(
            """UPDATE edges
               SET relation = ?,
                   weight = ?,
                   source = ?,
                   metadata = ?
               WHERE id = ?""",
            (
                edge.relation,
                edge.weight,
                edge.source,
                json.dumps(edge.metadata),
                edge.id,
            ),
        )
        self.conn.commit()
        return self.get_edge(edge_id)

    def delete_edge(self, edge_id: str) -> dict:
        cursor = self.conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        self.conn.commit()
        return {"edge_id": edge_id, "deleted": cursor.rowcount > 0}

    # --- Integrity audit helpers ---

    def find_dangling_edges(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS count
               FROM edges e
               LEFT JOIN knowledge_units from_unit ON from_unit.id = e.from_unit_id
               LEFT JOIN knowledge_units to_unit ON to_unit.id = e.to_unit_id
               WHERE from_unit.id IS NULL OR to_unit.id IS NULL"""
        ).fetchone()
        rows = self.conn.execute(
            """SELECT e.id, e.from_unit_id, e.to_unit_id, e.relation,
                      from_unit.id IS NULL AS missing_from,
                      to_unit.id IS NULL AS missing_to
               FROM edges e
               LEFT JOIN knowledge_units from_unit ON from_unit.id = e.from_unit_id
               LEFT JOIN knowledge_units to_unit ON to_unit.id = e.to_unit_id
               WHERE from_unit.id IS NULL OR to_unit.id IS NULL
               ORDER BY e.created_at DESC, e.id
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {
                    "edge_id": r["id"],
                    "from_unit_id": r["from_unit_id"],
                    "to_unit_id": r["to_unit_id"],
                    "relation": r["relation"],
                    "missing_from": bool(r["missing_from"]),
                    "missing_to": bool(r["missing_to"]),
                }
                for r in rows
            ],
        }

    def find_self_loop_edges(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM edges WHERE from_unit_id = to_unit_id"
        ).fetchone()
        rows = self.conn.execute(
            """SELECT id, from_unit_id, to_unit_id, relation
               FROM edges
               WHERE from_unit_id = to_unit_id
               ORDER BY created_at DESC, id
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {
                    "edge_id": r["id"],
                    "unit_id": r["from_unit_id"],
                    "relation": r["relation"],
                }
                for r in rows
            ],
        }

    def find_duplicate_edge_triples(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS count
               FROM (
                   SELECT 1
                   FROM edges
                   GROUP BY from_unit_id, to_unit_id, relation
                   HAVING COUNT(*) > 1
               )"""
        ).fetchone()
        rows = self.conn.execute(
            """SELECT from_unit_id, to_unit_id, relation, COUNT(*) AS duplicate_count,
                      json_group_array(id) AS edge_ids
               FROM edges
               GROUP BY from_unit_id, to_unit_id, relation
               HAVING COUNT(*) > 1
               ORDER BY duplicate_count DESC, from_unit_id, to_unit_id, relation
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {
                    "from_unit_id": r["from_unit_id"],
                    "to_unit_id": r["to_unit_id"],
                    "relation": r["relation"],
                    "duplicate_count": r["duplicate_count"],
                    "edge_ids": json.loads(r["edge_ids"]),
                }
                for r in rows
            ],
        }

    def find_units_missing_fts_rows(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS count
               FROM knowledge_units u
               LEFT JOIN knowledge_fts f ON f.unit_id = u.id
               WHERE f.unit_id IS NULL"""
        ).fetchone()
        rows = self.conn.execute(
            """SELECT u.id, u.title, u.source_project, u.source_entity_type
               FROM knowledge_units u
               LEFT JOIN knowledge_fts f ON f.unit_id = u.id
               WHERE f.unit_id IS NULL
               ORDER BY u.created_at DESC, u.id
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {
                    "unit_id": r["id"],
                    "title": r["title"],
                    "source_project": r["source_project"],
                    "source_entity_type": r["source_entity_type"],
                }
                for r in rows
            ],
        }

    def find_stale_fts_rows(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS count
               FROM knowledge_fts f
               LEFT JOIN knowledge_units u ON u.id = f.unit_id
               WHERE u.id IS NULL"""
        ).fetchone()
        rows = self.conn.execute(
            """SELECT f.rowid, f.unit_id, f.title
               FROM knowledge_fts f
               LEFT JOIN knowledge_units u ON u.id = f.unit_id
               WHERE u.id IS NULL
               ORDER BY f.rowid
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {"fts_rowid": r["rowid"], "unit_id": r["unit_id"], "title": r["title"]}
                for r in rows
            ],
        }

    def find_invalid_json_rows(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            """SELECT
                   (SELECT COUNT(*) FROM knowledge_units
                    WHERE NOT json_valid(metadata) OR NOT json_valid(tags))
                 + (SELECT COUNT(*) FROM edges WHERE NOT json_valid(metadata))
                 AS count"""
        ).fetchone()
        rows = self.conn.execute(
            """SELECT 'knowledge_units' AS table_name, id, NULL AS edge_id,
                      NOT json_valid(metadata) AS invalid_metadata,
                      NOT json_valid(tags) AS invalid_tags
               FROM knowledge_units
               WHERE NOT json_valid(metadata) OR NOT json_valid(tags)
               UNION ALL
               SELECT 'edges' AS table_name, NULL AS id, id AS edge_id,
                      NOT json_valid(metadata) AS invalid_metadata,
                      0 AS invalid_tags
               FROM edges
               WHERE NOT json_valid(metadata)
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {
                    "table": r["table_name"],
                    "unit_id": r["id"],
                    "edge_id": r["edge_id"],
                    "invalid_metadata": bool(r["invalid_metadata"]),
                    "invalid_tags": bool(r["invalid_tags"]),
                }
                for r in rows
            ],
        }

    def find_blank_units(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS count
               FROM knowledge_units
               WHERE trim(title) = '' OR trim(content) = ''"""
        ).fetchone()
        rows = self.conn.execute(
            """SELECT id, title, source_project, source_entity_type,
                      trim(title) = '' AS blank_title,
                      trim(content) = '' AS blank_content
               FROM knowledge_units
               WHERE trim(title) = '' OR trim(content) = ''
               ORDER BY created_at DESC, id
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {
                    "unit_id": r["id"],
                    "title": r["title"],
                    "source_project": r["source_project"],
                    "source_entity_type": r["source_entity_type"],
                    "blank_title": bool(r["blank_title"]),
                    "blank_content": bool(r["blank_content"]),
                }
                for r in rows
            ],
        }

    def repair_fts_index_integrity(self) -> dict:
        stale_cursor = self.conn.execute(
            """DELETE FROM knowledge_fts
               WHERE unit_id NOT IN (SELECT id FROM knowledge_units)"""
        )
        rows = self.conn.execute(
            """SELECT u.id, u.title, u.content, u.tags
               FROM knowledge_units u
               LEFT JOIN knowledge_fts f ON f.unit_id = u.id
               WHERE f.unit_id IS NULL"""
        ).fetchall()
        for row in rows:
            try:
                tags = json.loads(row["tags"])
            except json.JSONDecodeError:
                tags = []
            if not isinstance(tags, list):
                tags = []
            self.conn.execute(
                "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
                (row["id"], row["title"], row["content"], " ".join(map(str, tags))),
            )
        self.conn.commit()
        return {
            "fts_rows_inserted": len(rows),
            "fts_rows_deleted": stale_cursor.rowcount,
        }

    def get_backlinks(
        self,
        unit_id: str,
        *,
        direction: str = "both",
        relation: str | None = None,
        limit: int = 20,
    ) -> dict:
        center = self.get_unit(unit_id)
        if center is None:
            return {"center": None, "links": []}

        if direction not in ("incoming", "outgoing", "both"):
            raise ValueError("direction must be incoming, outgoing, or both")

        limit = max(0, limit)
        where = []
        params: list = []
        if direction in ("incoming", "both"):
            where.append("(e.to_unit_id = ?)")
            params.append(unit_id)
        if direction in ("outgoing", "both"):
            where.append("(e.from_unit_id = ?)")
            params.append(unit_id)

        query = f"""
            SELECT
                e.id AS edge_id,
                e.from_unit_id,
                e.to_unit_id,
                e.relation,
                e.weight,
                e.source,
                e.metadata AS edge_metadata,
                e.created_at AS edge_created_at,
                u.*
            FROM edges e
            JOIN knowledge_units u
              ON u.id = CASE
                  WHEN e.from_unit_id = ? THEN e.to_unit_id
                  ELSE e.from_unit_id
              END
            WHERE ({' OR '.join(where)})
        """
        query_params: list = [unit_id, *params]
        if relation:
            query += " AND e.relation = ?"
            query_params.append(relation)
        query += " ORDER BY e.created_at DESC, e.id LIMIT ?"
        query_params.append(limit)

        rows = self.conn.execute(query, query_params).fetchall()
        links = []
        for row in rows:
            edge = KnowledgeEdge(
                id=row["edge_id"],
                from_unit_id=row["from_unit_id"],
                to_unit_id=row["to_unit_id"],
                relation=row["relation"],
                weight=row["weight"],
                source=row["source"],
                metadata=json.loads(row["edge_metadata"]),
                created_at=row["edge_created_at"],
            )
            links.append(
                {
                    "direction": "outgoing"
                    if row["from_unit_id"] == unit_id
                    else "incoming",
                    "relation": str(edge.relation),
                    "edge": edge,
                    "unit": _row_to_unit(row),
                }
            )
        return {"center": center, "links": links}

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
                """SELECT unit_id,
                          rank,
                          snippet(knowledge_fts, -1, '[', ']', '...', 24) AS snippet
                   FROM knowledge_fts
                   WHERE knowledge_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit),
            ).fetchall()
            results = [
                {
                    "unit_id": r["unit_id"],
                    "rank": r["rank"],
                    "snippet": r["snippet"] or "",
                }
                for r in rows
            ]
            if _requires_exact_single_term_filter(query):
                exact = query.strip().lower()
                filtered = []
                for result in results:
                    unit = self.get_unit(result["unit_id"])
                    if unit is None:
                        continue
                    haystacks = [unit.title, unit.content, *unit.tags]
                    if any(exact in str(value).lower() for value in haystacks):
                        filtered.append(result)
                return filtered
            return results
        except sqlite3.OperationalError:
            # Fallback to LIKE search if FTS query syntax is invalid
            terms = _fallback_search_terms(query)
            clauses = " OR ".join(
                ["title LIKE ? OR content LIKE ? OR tags LIKE ?" for _ in terms]
            )
            params: list[object] = []
            for term in terms:
                pattern = f"%{term}%"
                params.extend([pattern, pattern, pattern])
            params.append(limit)
            rows = self.conn.execute(
                f"""SELECT id as unit_id, content, -1.0 as rank
                   FROM knowledge_units
                   WHERE {clauses}
                   LIMIT ?""",
                params,
            ).fetchall()
            return [
                {
                    "unit_id": r["unit_id"],
                    "rank": r["rank"],
                    "snippet": _excerpt(r["content"], query),
                }
                for r in rows
            ]

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
