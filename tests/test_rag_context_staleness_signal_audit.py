from __future__ import annotations

from graph.rag.context_staleness_signal import analyze_context_staleness_signals


def test_context_staleness_signals_bucket_dates_against_fixed_reference():
    report = analyze_context_staleness_signals(
        [
            {"id": "recent", "date": "2024-12-15"},
            {"id": "stale", "date": "2022-01-01"},
            {"id": "missing"},
            {"id": "invalid", "date": "not-a-date"},
            {"id": "future", "date": "2025-02-01"},
        ],
        reference_date="2025-01-01",
    )

    assert report["freshness_buckets"] == {"recent": 1, "stale": 1, "missing_date": 1, "invalid_date": 1, "future_date": 1}
    assert report["stale_item_ids"] == ["stale"]
    assert report["missing_date_item_ids"] == ["missing"]
    assert report["invalid_date_item_ids"] == ["invalid"]
    assert report["future_date_item_ids"] == ["future"]
