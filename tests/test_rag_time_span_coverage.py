from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graph.rag.time_span_coverage import analyze_time_span_coverage
from graph.types.models import KnowledgeUnit


NOW = datetime(2026, 5, 12, tzinfo=timezone.utc)


def test_analyze_time_span_coverage_reports_year_buckets_and_undated_results():
    payload = analyze_time_span_coverage(
        [
            {"id": "old", "source_project": "notes", "created_at": "2024-01-01"},
            {"id": "new", "source_project": "web", "metadata": {"published_at": "2026-05-01"}},
            {"id": "undated", "source_project": "notes"},
        ],
        now=NOW,
    )

    assert payload["totals"] == {
        "result_count": 3,
        "dated_count": 2,
        "undated_count": 1,
        "bucket": "year",
    }
    assert payload["earliest_timestamp"] == "2024-01-01T00:00:00+00:00"
    assert payload["latest_timestamp"] == "2026-05-01T00:00:00+00:00"
    assert payload["bucket_counts"] == [
        {"bucket": "2024", "count": 1},
        {"bucket": "2026", "count": 1},
    ]
    assert payload["source_distribution"] == [
        {"source": "notes", "count": 2},
        {"source": "web", "count": 1},
    ]
    assert [row["result_id"] for row in payload["representative_rows"]] == ["old", "new", "undated"]


def test_analyze_time_span_coverage_supports_month_buckets_and_nested_shapes():
    unit = KnowledgeUnit(
        id="unit-1",
        source_project="readwise",
        source_id="source-1",
        source_entity_type="highlight",
        title="Highlight",
        content="Text",
        metadata={"published_at": "2026-04-02"},
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 3, tzinfo=timezone.utc),
    )
    wrapper = SimpleNamespace(id="wrapper", unit=unit)

    payload = analyze_time_span_coverage(
        [
            wrapper,
            ({"source_id": "tuple", "source": "archive", "date": "2026-05-05"}, 0.8),
        ],
        now=NOW,
        bucket="month",
    )

    assert payload["bucket_counts"] == [
        {"bucket": "2026-04", "count": 1},
        {"bucket": "2026-05", "count": 1},
    ]
    assert payload["bucket_source_distribution"] == {
        "2026-04": [{"source": "readwise", "count": 1}],
        "2026-05": [{"source": "archive", "count": 1}],
    }
    assert payload["representative_rows"][0]["result_id"] == "wrapper"
    assert payload["representative_rows"][0]["age_days"] == 39


def test_analyze_time_span_coverage_limits_representatives_deterministically():
    payload = analyze_time_span_coverage(
        [
            {"id": "b", "source_project": "notes", "date": "2026-01-02"},
            {"id": "a", "source_project": "notes", "date": "2026-01-01"},
            {"id": "c", "source_project": "notes", "date": "2026-01-03"},
        ],
        now=NOW,
        limit=2,
    )

    assert [row["result_id"] for row in payload["representative_rows"]] == ["a", "b"]


@pytest.mark.parametrize("kwargs", [{"bucket": "week"}, {"limit": 0}, {"limit": True}])
def test_analyze_time_span_coverage_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        analyze_time_span_coverage([], **kwargs)
