from __future__ import annotations

from types import SimpleNamespace

from graph.rag.result_staleness_buckets import bucket_results_by_staleness


def test_result_staleness_buckets_handles_mixed_result_shapes():
    report = bucket_results_by_staleness(
        [
            {"id": "fresh", "published_at": "2026-04-20"},
            ({"id": "recent", "metadata": {"updated_at": "2026-03-01"}}, 0.5),
            SimpleNamespace(id="aging", metadata={"date": "2026-01-15"}),
            {"id": "stale", "created_at": "2025-01-01"},
            {"id": "undated"},
        ],
        now="2026-05-01",
    )

    assert report["bucket_counts"] == {"fresh": 1, "recent": 1, "aging": 1, "stale": 1, "undated": 1}
    assert report["result_ids"]["fresh"] == ["fresh"]
    assert report["oldest_date"] == "2025-01-01"
    assert report["newest_date"] == "2026-04-20"
    assert report["warnings"] == []


def test_result_staleness_buckets_warns_for_empty_and_undated():
    assert bucket_results_by_staleness([])["warnings"] == ["no_results"]
    assert bucket_results_by_staleness([{"id": "a"}, {"id": "b", "date": ""}])["warnings"] == ["undated_results_dominate"]
