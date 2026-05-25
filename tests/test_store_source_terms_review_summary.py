from __future__ import annotations

from graph.store.source_terms_review_summary import summarize_source_terms_reviews


def test_summarize_source_terms_reviews_counts_stale_and_missing_reviews():
    summary = summarize_source_terms_reviews(
        [
            {"host": "current.test", "terms_reviewed_at": "2026-05-01T00:00:00+00:00"},
            {"url": "https://stale-b.test/terms", "metadata": {"reviewed_at": "2025-01-01T00:00:00+00:00"}},
            {"host": "stale-a.test", "metadata": {"terms_reviewed_at": "2024-12-15T00:00:00+00:00"}},
            {"host": "missing.test", "metadata": {}},
        ],
        reference_date="2026-05-26T00:00:00+00:00",
        max_age_days=365,
    )

    assert summary == {
        "reviewed_sources": 3,
        "stale_review_count": 2,
        "missing_review_count": 1,
        "average_days_since_review": 354.0,
        "stale_hosts": ["stale-a.test", "stale-b.test"],
    }


def test_summarize_source_terms_reviews_all_missing_has_zero_average():
    summary = summarize_source_terms_reviews(
        [{"host": "missing.test"}],
        reference_date="2026-05-26T00:00:00+00:00",
    )

    assert summary["reviewed_sources"] == 0
    assert summary["average_days_since_review"] == 0.0
