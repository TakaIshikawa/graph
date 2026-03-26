"""Adapter for the forty-two project (experiment knowledge graph)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class FortyTwoAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "forty_two"

    @property
    def entity_types(self) -> list[str]:
        return ["knowledge_node", "knowledge_edge"]

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

        if "knowledge_node" in types:
            self._ingest_nodes(conn, result, since)

        if "knowledge_edge" in types:
            self._ingest_edges(conn, result)

        conn.close()
        return result

    def _ingest_nodes(
        self, conn: sqlite3.Connection, result: IngestResult, since: SyncState | None
    ) -> None:
        where = ""
        params: list = []
        if since and since.last_sync_at:
            where = "WHERE kn.created_at > ?"
            params.append(
                since.last_sync_at.isoformat()
                if hasattr(since.last_sync_at, "isoformat")
                else str(since.last_sync_at)
            )

        rows = conn.execute(
            f"""SELECT kn.id, kn.summary, kn.utility_contribution,
                       kn.tags, kn.findings, kn.is_negative, kn.novelty_score,
                       kn.created_at,
                       e.title AS experiment_title, e.hypothesis
                FROM knowledge_nodes kn
                LEFT JOIN experiments e ON kn.experiment_id = e.id
                {where}
                ORDER BY kn.created_at""",
            params,
        ).fetchall()

        for row in rows:
            tags_raw = row["tags"]
            tags = json.loads(tags_raw) if tags_raw else []
            findings_raw = row["findings"]
            findings = json.loads(findings_raw) if findings_raw else {}

            title = row["experiment_title"] or (row["summary"] or "")[:80]
            content = row["summary"] or ""

            unit = KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id=row["id"],
                source_entity_type="knowledge_node",
                title=title,
                content=content,
                content_type=ContentType.FINDING,
                metadata={
                    "hypothesis": row["hypothesis"],
                    "findings": findings,
                    "is_negative": bool(row["is_negative"]),
                    "novelty_score": row["novelty_score"],
                },
                tags=tags,
                utility_score=row["utility_contribution"],
                created_at=row["created_at"] or "1970-01-01T00:00:00+00:00",
            )
            result.units.append(unit)

    def _ingest_edges(self, conn: sqlite3.Connection, result: IngestResult) -> None:
        rows = conn.execute(
            "SELECT id, from_node_id, to_node_id, relation, weight FROM knowledge_edges"
        ).fetchall()

        for row in rows:
            relation_str = row["relation"]
            try:
                relation = EdgeRelation(relation_str)
            except ValueError:
                relation = EdgeRelation.RELATES_TO

            edge = KnowledgeEdge(
                from_unit_id=row["from_node_id"],
                to_unit_id=row["to_node_id"],
                relation=relation,
                weight=row["weight"] or 1.0,
                source=EdgeSource.SOURCE,
                metadata={"original_id": row["id"], "source_project": "forty_two"},
            )
            result.edges.append(edge)
