"""Recommend a source-diverse result order for answer drafting."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import number, result_id, source_id, value


def plan_result_source_rotation(results: Iterable[Any]) -> dict[str, Any]:
    """Reorder results to reduce adjacent repeated sources while preserving score order."""
    remaining = [_row(record, index) for index, record in enumerate(results)]
    ordered: list[dict[str, Any]] = []
    while remaining:
        previous_source = ordered[-1]["source_id"] if ordered else None
        candidates = [row for row in remaining if row["source_id"] != previous_source]
        pool = candidates or remaining
        chosen = min(pool, key=lambda row: row["original_rank"])
        ordered.append(chosen)
        remaining.remove(chosen)

    plan = []
    for new_rank, row in enumerate(ordered, start=1):
        moved = row["original_rank"] != new_rank
        plan.append(
            {
                "result_id": row["result_id"],
                "source_id": row["source_id"],
                "original_rank": row["original_rank"],
                "recommended_rank": new_rank,
                "score": row["score"],
                "movement_reason": "source_diversity" if moved else "preserve_score_order",
            }
        )
    return {"rotation": plan}


def _row(record: Any, index: int) -> dict[str, Any]:
    return {
        "result_id": result_id(record, index),
        "source_id": source_id(record) or "unknown",
        "score": number(value(record, "score")) or 0.0,
        "original_rank": index + 1,
    }
