"""Adapter for the max project (idea generation engine)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

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
        buildable_columns = _columns(conn, "buildable_units")
        has_feedback = _table_exists(conn, "feedback")
        has_evaluations = _table_exists(conn, "evaluations")
        has_critiques = _table_exists(conn, "idea_critiques")

        where = ""
        params: list = []
        if since and since.last_sync_at:
            since_value = (
                since.last_sync_at.isoformat()
                if hasattr(since.last_sync_at, "isoformat")
                else str(since.last_sync_at)
            )
            changed = ["b.updated_at > ?"]
            params.append(since_value)
            if has_feedback:
                changed.append("lf.created_at > ?")
                params.append(since_value)
            if has_critiques:
                changed.append("lc.created_at > ?")
                params.append(since_value)
            where = f"WHERE {' OR '.join(changed)}"

        select_fields = [
            "b.id",
            "b.title",
            "b.one_liner",
            "b.category",
            "b.problem",
            "b.solution",
            "b.tech_approach",
            "b.inspiring_insights",
            "b.status",
            "b.created_at",
            "b.updated_at",
        ]
        for column in [
            "domain",
            "target_users",
            "specific_user",
            "buyer",
            "workflow_context",
            "current_workaround",
            "why_now",
            "validation_plan",
            "first_10_customers",
            "domain_risks",
            "evidence_rationale",
            "novelty_score",
            "usefulness_score",
            "quality_score",
            "rejection_tags",
        ]:
            select_fields.append(_select_expr(buildable_columns, column, alias=column))

        joins = []
        ctes = []
        if has_feedback:
            ctes.append(
                """latest_feedback AS (
                    SELECT f.*
                    FROM feedback f
                    JOIN (
                        SELECT buildable_unit_id, MAX(created_at) AS created_at
                        FROM feedback
                        GROUP BY buildable_unit_id
                    ) latest
                      ON latest.buildable_unit_id = f.buildable_unit_id
                     AND latest.created_at = f.created_at
                )"""
            )
            joins.append("LEFT JOIN latest_feedback lf ON lf.buildable_unit_id = b.id")
            select_fields.extend(
                [
                    "lf.outcome AS feedback_outcome",
                    "lf.reason AS feedback_reason",
                    "lf.created_at AS reviewed_at",
                    _column_expr(conn, "feedback", "approval_score", "NULL", "approval_score", "lf"),
                ]
            )
        else:
            select_fields.extend(
                [
                    "NULL AS feedback_outcome",
                    "NULL AS feedback_reason",
                    "NULL AS reviewed_at",
                    "NULL AS approval_score",
                ]
            )

        if has_evaluations:
            joins.append("LEFT JOIN evaluations e ON e.buildable_unit_id = b.id")
            select_fields.extend(
                [
                    _column_expr(conn, "evaluations", "overall_score", "NULL", "overall_score", "e"),
                    _column_expr(conn, "evaluations", "recommendation", "NULL", "recommendation", "e"),
                ]
            )
        else:
            select_fields.extend(["NULL AS overall_score", "NULL AS recommendation"])

        if has_critiques:
            ctes.append(
                """latest_critique AS (
                    SELECT c.*
                    FROM idea_critiques c
                    JOIN (
                        SELECT buildable_unit_id, MAX(created_at) AS created_at
                        FROM idea_critiques
                        GROUP BY buildable_unit_id
                    ) latest
                      ON latest.buildable_unit_id = c.buildable_unit_id
                     AND latest.created_at = c.created_at
                )"""
            )
            joins.append("LEFT JOIN latest_critique lc ON lc.buildable_unit_id = b.id")
            select_fields.extend(
                [
                    "lc.dimensions AS critique_dimensions",
                    "lc.reasoning AS critique_reasoning",
                    "lc.rejection_tags AS critique_rejection_tags",
                    "lc.evidence_pack AS evidence_pack",
                    "lc.created_at AS critiqued_at",
                ]
            )
        else:
            select_fields.extend(
                [
                    "NULL AS critique_dimensions",
                    "NULL AS critique_reasoning",
                    "NULL AS critique_rejection_tags",
                    "NULL AS evidence_pack",
                    "NULL AS critiqued_at",
                ]
            )

        with_clause = f"WITH {', '.join(ctes)} " if ctes else ""
        rows = conn.execute(
            f"""{with_clause}
                SELECT {', '.join(select_fields)}
                FROM buildable_units b
                {' '.join(joins)}
                {where}
                ORDER BY b.updated_at""",
            params,
        ).fetchall()

        for row in rows:
            inspiring_raw = row["inspiring_insights"]
            inspiring = _json_value(inspiring_raw, [])

            content = row["one_liner"] or ""
            if row["problem"]:
                content += f"\n\nProblem: {row['problem']}"
            if row["solution"]:
                content += f"\nSolution: {row['solution']}"
            if row["specific_user"]:
                content += f"\nSpecific user: {row['specific_user']}"
            if row["buyer"]:
                content += f"\nBuyer: {row['buyer']}"
            if row["workflow_context"]:
                content += f"\nWorkflow: {row['workflow_context']}"
            if row["validation_plan"]:
                content += f"\nValidation: {row['validation_plan']}"

            review_state = _review_state(row["status"], row["feedback_outcome"])
            labels = _graph_labels(review_state)
            is_approved = review_state == "approved"
            tags = _idea_tags(row, review_state, is_approved)
            critique_rejection_tags = _json_value(row["critique_rejection_tags"], [])
            rejection_tags = _json_value(row["rejection_tags"], [])
            merged_rejection_tags = sorted(
                {str(tag) for tag in [*rejection_tags, *critique_rejection_tags] if tag}
            )
            overall_score = row["overall_score"]
            quality_score = row["quality_score"]
            utility_score = None
            if overall_score is not None:
                utility_score = float(overall_score) / 100.0
            elif quality_score is not None:
                utility_score = float(quality_score) / 10.0

            unit = KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id=row["id"],
                source_entity_type="buildable_unit",
                title=row["title"],
                content=content,
                content_type=ContentType.IDEA,
                metadata={
                    "category": row["category"],
                    "domain": row["domain"],
                    "target_users": row["target_users"],
                    "specific_user": row["specific_user"],
                    "buyer": row["buyer"],
                    "workflow_context": row["workflow_context"],
                    "current_workaround": row["current_workaround"],
                    "why_now": row["why_now"],
                    "validation_plan": row["validation_plan"],
                    "first_10_customers": row["first_10_customers"],
                    "domain_risks": _json_value(row["domain_risks"], []),
                    "evidence_rationale": row["evidence_rationale"],
                    "tech_approach": row["tech_approach"],
                    "status": row["status"],
                    "review_state": review_state,
                    "feedback_outcome": row["feedback_outcome"],
                    "feedback_reason": row["feedback_reason"],
                    "reviewed_at": row["reviewed_at"],
                    "approval_score": row["approval_score"],
                    "is_approved": is_approved,
                    "graph_labels": labels,
                    "quality_score": quality_score,
                    "novelty_score": row["novelty_score"],
                    "usefulness_score": row["usefulness_score"],
                    "overall_score": overall_score,
                    "recommendation": row["recommendation"],
                    "rejection_tags": merged_rejection_tags,
                    "critique_dimensions": _json_value(row["critique_dimensions"], {}),
                    "critique_reasoning": row["critique_reasoning"],
                    "evidence_pack": _json_value(row["evidence_pack"], {}),
                    "critiqued_at": row["critiqued_at"],
                },
                tags=tags,
                utility_score=utility_score,
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


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _select_expr(columns: set[str], column: str, *, alias: str) -> str:
    if column in columns:
        return f"b.{column} AS {alias}"
    return f"NULL AS {alias}"


def _column_expr(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    default_sql: str,
    alias: str,
    table_alias: str,
) -> str:
    if column in _columns(conn, table):
        return f"{table_alias}.{column} AS {alias}"
    return f"{default_sql} AS {alias}"


def _json_value(raw: Any, default: Any) -> Any:
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def _review_state(status: str | None, feedback_outcome: str | None) -> str:
    outcome = (feedback_outcome or "").strip().lower()
    if outcome in {"approved", "rejected"}:
        return outcome
    if outcome:
        return f"feedback_{outcome}"

    normalized_status = (status or "draft").strip().lower()
    if normalized_status in {"approved", "rejected"}:
        return normalized_status
    if normalized_status in {"evaluated", "draft", "generated"}:
        return "unreviewed"
    return normalized_status or "unreviewed"


def _graph_labels(review_state: str) -> list[str]:
    suffix = "".join(part.capitalize() for part in review_state.replace("-", "_").split("_"))
    return ["Idea", f"Review{suffix}"]


def _idea_tags(row: sqlite3.Row, review_state: str, is_approved: bool) -> list[str]:
    tags = []
    for value in [row["category"], row["domain"], f"review:{review_state}"]:
        if value:
            tags.append(str(value))
    if is_approved:
        tags.append("approved")
    if row["feedback_outcome"]:
        tags.append(f"feedback:{row['feedback_outcome']}")
    if row["recommendation"]:
        tags.append(f"recommendation:{row['recommendation']}")
    return list(dict.fromkeys(tags))
