from __future__ import annotations

from graph.store.saved_query_usage_summary import saved_query_usage_summary


class Query:
    def __init__(self, metadata: dict):
        self.metadata = metadata


def test_saved_query_usage_summary_aggregates_by_type_or_source():
    rows = saved_query_usage_summary(
        [
            {"query_type": "search", "run_count": 2, "last_run_at": "2026-05-20T00:00:00Z", "result_count": 4},
            {"query_type": "search", "run_count": 1, "last_run_at": "2026-05-22T00:00:00Z", "result_count": 0},
            Query({"source": "docs", "runs": "3", "last_executed_at": "2026-04-01", "last_result_count": "10"}),
            {"source_project": "docs", "run_count": 1, "zero_result_runs": 1},
        ],
        reference_date="2026-05-27T00:00:00+00:00",
        stale_after_days=30,
    )

    assert rows == [
        {
            "query_type": "docs",
            "run_count": 4,
            "last_run_at": "2026-04-01T00:00:00+00:00",
            "average_result_count": 10.0,
            "zero_result_runs": 1,
            "stale_query": True,
        },
        {
            "query_type": "search",
            "run_count": 3,
            "last_run_at": "2026-05-22T00:00:00+00:00",
            "average_result_count": 2.0,
            "zero_result_runs": 1,
            "stale_query": False,
        },
    ]
