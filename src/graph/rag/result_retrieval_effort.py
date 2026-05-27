"""Summarize retrieval effort metadata across RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag._analysis_utils import number, result_id, string, value

_LOW_SCORE_THRESHOLD = 0.5


def summarize_result_retrieval_effort(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return deterministic aggregate effort metadata for result dictionaries."""
    rows = list(results or [])
    pass_counts: Counter[str] = Counter()
    query_variant_counts: Counter[str] = Counter()
    source_project_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    latencies: list[float] = []
    total_tokens = 0
    scored_rows: list[tuple[float, int, str]] = []

    for index, result in enumerate(rows):
        rank = _rank(result, index)
        rid = result_id(result, index)

        retrieval_pass = string(value(result, "retrieval_pass"))
        if retrieval_pass is None:
            missing_counts["retrieval_pass"] += 1
        else:
            pass_counts[retrieval_pass] += 1

        query_variant = string(value(result, "query_variant"))
        if query_variant is None:
            missing_counts["query_variant"] += 1
        else:
            query_variant_counts[query_variant] += 1

        source_project = string(value(result, "source_project"))
        if source_project is None:
            missing_counts["source_project"] += 1
        else:
            source_project_counts[source_project] += 1

        latency = number(value(result, "latency_ms"))
        if latency is None:
            missing_counts["latency_ms"] += 1
        else:
            latencies.append(latency)

        token_count = number(value(result, "token_count"))
        if token_count is None:
            missing_counts["token_count"] += 1
        else:
            total_tokens += int(token_count)

        score = number(value(result, "score"))
        if score is None:
            missing_counts["score"] += 1
        else:
            scored_rows.append((score, rank, rid))

    low_score_tail_ids = [
        rid for score, _, rid in sorted(scored_rows, key=lambda item: (item[0], item[1], item[2])) if score < _LOW_SCORE_THRESHOLD
    ]
    return {
        "result_count": len(rows),
        "pass_counts": dict(sorted(pass_counts.items())),
        "query_variant_counts": dict(sorted(query_variant_counts.items())),
        "source_project_counts": dict(sorted(source_project_counts.items())),
        "average_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "total_tokens": total_tokens,
        "low_score_tail_ids": low_score_tail_ids,
        "missing_metadata_counts": dict(sorted(missing_counts.items())),
    }


def _rank(result: Any, index: int) -> int:
    rank = number(value(result, "rank"))
    return int(rank) if rank is not None else index + 1
