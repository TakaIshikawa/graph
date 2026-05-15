from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graph.rag.result_timeline_gaps import analyze_result_timeline_gaps
from graph.types.models import KnowledgeUnit


def test_analyze_result_timeline_gaps_reports_large_gaps_and_year_buckets():
    report = analyze_result_timeline_gaps(
        [
            {"id": "middle", "metadata": {"created_at": "2025-02-01"}},
            {"id": "old", "published_at": "2024-01-01T00:30:00+09:00"},
            {"id": "next", "updated_at": datetime(2025, 2, 15, tzinfo=timezone.utc)},
            {"id": "undated", "title": "No date"},
            {"id": "latest", "date": "2025-06-01"},
        ],
        gap_days=90,
    )

    assert report == {
        "result_count": 5,
        "dated_count": 4,
        "undated_count": 1,
        "earliest_date": "2023-12-31",
        "latest_date": "2025-06-01",
        "year_buckets": [
            {"year": "2023", "count": 1},
            {"year": "2025", "count": 3},
        ],
        "gaps": [
            {
                "previous_result_id": "old",
                "next_result_id": "middle",
                "previous_date": "2023-12-31",
                "next_date": "2025-02-01",
                "gap_days": 398,
            },
            {
                "previous_result_id": "next",
                "next_result_id": "latest",
                "previous_date": "2025-02-15",
                "next_date": "2025-06-01",
                "gap_days": 106,
            },
        ],
    }


def test_analyze_result_timeline_gaps_supports_nested_knowledge_unit_results():
    unit = KnowledgeUnit(
        id="unit-1",
        source_project="readwise",
        source_id="source-1",
        source_entity_type="highlight",
        title="Highlight",
        content="Text",
        metadata={"published_at": "2025-04-01T01:00:00+09:00"},
    )
    wrapper = SimpleNamespace(id="wrapper", unit=unit)

    report = analyze_result_timeline_gaps(
        [
            {"source_id": "first", "created_at": "2025-01-01"},
            wrapper,
            ({"result_id": "tuple", "metadata": {"date": "2025-04-20"}}, 0.8),
            {"source_id": "empty", "metadata": {"published_at": "not-a-date"}},
        ],
        gap_days=88,
    )

    assert report["result_count"] == 4
    assert report["dated_count"] == 3
    assert report["undated_count"] == 1
    assert report["earliest_date"] == "2025-01-01"
    assert report["latest_date"] == "2025-04-20"
    assert report["year_buckets"] == [{"year": "2025", "count": 3}]
    assert report["gaps"] == [
        {
            "previous_result_id": "first",
            "next_result_id": "wrapper",
            "previous_date": "2025-01-01",
            "next_date": "2025-03-31",
            "gap_days": 89,
        }
    ]


def test_analyze_result_timeline_gaps_uses_strict_gap_threshold():
    report = analyze_result_timeline_gaps(
        [
            {"id": "a", "date": "2025-01-01"},
            {"id": "b", "date": "2025-04-01"},
        ],
        gap_days=90,
    )

    assert report["gaps"] == []


def test_analyze_result_timeline_gaps_handles_empty_results():
    assert analyze_result_timeline_gaps([]) == {
        "result_count": 0,
        "dated_count": 0,
        "undated_count": 0,
        "earliest_date": None,
        "latest_date": None,
        "year_buckets": [],
        "gaps": [],
    }


@pytest.mark.parametrize("gap_days", [-1, 1.5, "90", True])
def test_analyze_result_timeline_gaps_validates_gap_days(gap_days):
    with pytest.raises(ValueError, match="gap_days must be a non-negative integer"):
        analyze_result_timeline_gaps([], gap_days=gap_days)
