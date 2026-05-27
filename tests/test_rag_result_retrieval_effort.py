from __future__ import annotations

from graph.rag.result_retrieval_effort import summarize_result_retrieval_effort


def test_summarize_result_retrieval_effort_aggregates_metadata():
    summary = summarize_result_retrieval_effort(
        [
            {
                "id": "a",
                "rank": 2,
                "score": 0.42,
                "source_project": "docs",
                "query_variant": "original",
                "retrieval_pass": "semantic",
                "token_count": 120,
                "latency_ms": 20,
            },
            {
                "id": "b",
                "rank": 1,
                "score": 0.91,
                "source_project": "docs",
                "query_variant": "expanded",
                "retrieval_pass": "keyword",
                "token_count": 80,
                "latency_ms": 40,
            },
            {
                "id": "c",
                "rank": 3,
                "score": 0.42,
                "source_project": "notes",
                "query_variant": "original",
                "retrieval_pass": "semantic",
                "token_count": 50,
                "latency_ms": 30,
            },
        ]
    )

    assert summary == {
        "result_count": 3,
        "pass_counts": {"keyword": 1, "semantic": 2},
        "query_variant_counts": {"expanded": 1, "original": 2},
        "source_project_counts": {"docs": 2, "notes": 1},
        "average_latency_ms": 30.0,
        "total_tokens": 250,
        "low_score_tail_ids": ["a", "c"],
        "missing_metadata_counts": {},
    }


def test_summarize_result_retrieval_effort_tolerates_missing_metadata_and_zero_safe():
    summary = summarize_result_retrieval_effort([{"id": "missing"}, {"id": "nested", "metadata": {"score": 0.2}}])

    assert summary["average_latency_ms"] == 0.0
    assert summary["total_tokens"] == 0
    assert summary["low_score_tail_ids"] == ["nested"]
    assert summary["missing_metadata_counts"] == {
        "latency_ms": 2,
        "query_variant": 2,
        "retrieval_pass": 2,
        "score": 1,
        "source_project": 2,
        "token_count": 2,
    }
