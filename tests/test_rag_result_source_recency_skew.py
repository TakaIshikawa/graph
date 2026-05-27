from __future__ import annotations

from graph.rag.result_source_recency_skew import analyze_result_source_recency_skew


class Result:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_result_source_recency_skew_groups_dates_and_flags_stale_source():
    summary = analyze_result_source_recency_skew(
        [
            {"source": "Fresh", "published_at": "2024-01-01"},
            {"source": "Fresh", "metadata": {"updated_at": "2024-01-11"}},
            {"source": "Archive", "date": "2023-01-01"},
            {"source": "Archive", "created_at": "2023-02-01"},
            {"source": "Fresh"},
        ],
        reference_date="2024-02-01",
    )

    assert summary["total_results"] == 5
    assert summary["dated_count"] == 4
    assert summary["undated_count"] == 1
    assert summary["overall_median_age_days"] == 198.0
    assert summary["source_summaries"] == [
        {
            "source": "archive",
            "result_count": 2,
            "newest_date": "2023-02-01",
            "oldest_date": "2023-01-01",
            "median_age_days": 380.5,
            "stale_share": 1.0,
        },
        {
            "source": "fresh",
            "result_count": 2,
            "newest_date": "2024-01-11",
            "oldest_date": "2024-01-01",
            "median_age_days": 26.0,
            "stale_share": 0.0,
        },
    ]
    assert [row["source"] for row in summary["skewed_sources"]] == ["archive"]


def test_result_source_recency_skew_uses_provider_domain_and_object_inputs():
    summary = analyze_result_source_recency_skew(
        [
            Result(provider="api", metadata={"published_at": "2024-01-01"}),
            {"url": "https://Example.com/a", "metadata": {"date": "2024-01-15"}},
        ],
        reference_date="2024-02-01",
    )

    assert [row["source"] for row in summary["source_summaries"]] == ["api", "example.com"]
