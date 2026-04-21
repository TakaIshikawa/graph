"""Adapter for SOTA research papers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class SOTAAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "sota"

    @property
    def entity_types(self) -> list[str]:
        return ["paper"]

    def __init__(self, db_path: str = ""):
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

        # Only ingest breakthrough papers (is_breakthrough = 1)
        where_parts = ["is_breakthrough = 1"]
        params: list = []

        if since and since.last_sync_at:
            # Sync papers that were reviewed after last sync
            where_parts.append("reviewed_at > ?")
            params.append(
                since.last_sync_at.isoformat()
                if hasattr(since.last_sync_at, "isoformat")
                else str(since.last_sync_at)
            )

        where_clause = " AND ".join(where_parts)

        rows = conn.execute(
            f"""
            SELECT arxiv_id, title, abstract, authors, categories,
                   published_date, github_url, github_stars,
                   citation_count, influential_citations,
                   sota_results, breakthrough_score, score_breakdown,
                   tags, notes, reviewed_at
            FROM papers
            WHERE {where_clause}
            ORDER BY reviewed_at
            """,
            params,
        ).fetchall()

        for row in rows:
            source_id = f"paper_{row['arxiv_id']}"

            # Parse JSON fields
            authors = json.loads(row["authors"])
            categories = json.loads(row["categories"])
            tags = json.loads(row["tags"] or "[]")
            score_breakdown = json.loads(row["score_breakdown"] or "{}")

            # Build content with abstract + key metrics
            content_parts = [row["abstract"]]

            if row["citation_count"]:
                content_parts.append(f"\n\nCitations: {row['citation_count']}")

            if row["github_url"]:
                stars = f" ({row['github_stars']} ⭐)" if row["github_stars"] else ""
                content_parts.append(f"Code: {row['github_url']}{stars}")

            content = "".join(content_parts)

            # Build metadata
            metadata = {
                "arxiv_id": row["arxiv_id"],
                "authors": authors,
                "categories": categories,
                "published_date": row["published_date"],
                "arxiv_url": f"https://arxiv.org/abs/{row['arxiv_id']}",
                "breakthrough_score": row["breakthrough_score"],
                "score_breakdown": score_breakdown,
            }

            if row["github_url"]:
                metadata["github_url"] = row["github_url"]
                metadata["github_stars"] = row["github_stars"]

            if row["citation_count"]:
                metadata["citation_count"] = row["citation_count"]

            if row["influential_citations"]:
                metadata["influential_citations"] = row["influential_citations"]

            # Add primary category as tag
            primary_category = categories[0] if categories else "unknown"
            all_tags = tags + [primary_category]

            unit = KnowledgeUnit(
                source_project=SourceProject.SOTA,
                source_id=source_id,
                source_entity_type="paper",
                title=row["title"],
                content=content,
                content_type=ContentType.FINDING,
                metadata=metadata,
                tags=all_tags,
                created_at=row["published_date"],
            )

            result.units.append(unit)

        conn.close()
        return result
