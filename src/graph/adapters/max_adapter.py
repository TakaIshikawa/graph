"""Adapter for the max project (idea generation engine)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class MaxAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "max"

    @property
    def entity_types(self) -> list[str]:
        return ["insight", "buildable_unit"]

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

        if "insight" in types:
            self._ingest_insights(conn, result, since)

        if "buildable_unit" in types:
            self._ingest_buildable_units(conn, result, since)

        conn.close()
        return result

    def _ingest_insights(
        self, conn: sqlite3.Connection, result: IngestResult, since: SyncState | None
    ) -> None:
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
            f"""SELECT id, category, title, summary, evidence, confidence,
                       domains, implications, time_horizon, created_at
                FROM insights {where}
                ORDER BY created_at""",
            params,
        ).fetchall()

        for row in rows:
            domains = json.loads(row["domains"]) if row["domains"] else []
            evidence = json.loads(row["evidence"]) if row["evidence"] else []
            implications = json.loads(row["implications"]) if row["implications"] else []

            unit = KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id=row["id"],
                source_entity_type="insight",
                title=row["title"],
                content=row["summary"],
                content_type=ContentType.INSIGHT,
                metadata={
                    "category": row["category"],
                    "evidence": evidence,
                    "domains": domains,
                    "implications": implications,
                    "time_horizon": row["time_horizon"],
                },
                tags=domains,
                confidence=row["confidence"],
                created_at=row["created_at"] or "1970-01-01T00:00:00+00:00",
            )
            result.units.append(unit)

    def _ingest_buildable_units(
        self, conn: sqlite3.Connection, result: IngestResult, since: SyncState | None
    ) -> None:
        where = ""
        params: list = []
        if since and since.last_sync_at:
            where = "WHERE updated_at > ?"
            params.append(
                since.last_sync_at.isoformat()
                if hasattr(since.last_sync_at, "isoformat")
                else str(since.last_sync_at)
            )

        rows = conn.execute(
            f"""SELECT id, title, one_liner, category, problem, solution,
                       tech_approach, inspiring_insights, status, created_at, updated_at
                FROM buildable_units {where}
                ORDER BY updated_at""",
            params,
        ).fetchall()

        for row in rows:
            inspiring_raw = row["inspiring_insights"]
            inspiring = json.loads(inspiring_raw) if inspiring_raw else []

            content = row["one_liner"] or ""
            if row["problem"]:
                content += f"\n\nProblem: {row['problem']}"
            if row["solution"]:
                content += f"\nSolution: {row['solution']}"

            unit = KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id=row["id"],
                source_entity_type="buildable_unit",
                title=row["title"],
                content=content,
                content_type=ContentType.IDEA,
                metadata={
                    "category": row["category"],
                    "tech_approach": row["tech_approach"],
                    "status": row["status"],
                },
                tags=[row["category"]] if row["category"] else [],
                created_at=row["created_at"] or "1970-01-01T00:00:00+00:00",
            )
            result.units.append(unit)

            # Create inspires edges from insights to this buildable unit
            for insight_id in inspiring:
                edge = KnowledgeEdge(
                    from_unit_id=insight_id,
                    to_unit_id=row["id"],
                    relation=EdgeRelation.INSPIRES,
                    source=EdgeSource.SOURCE,
                    metadata={"source_project": "max"},
                )
                result.edges.append(edge)
