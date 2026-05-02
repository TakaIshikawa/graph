from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

import pytest

from graph.rag import build_source_timeline


@dataclass
class UnitStub:
    source_project: str
    created_at: object


@dataclass
class ResultStub:
    unit: UnitStub


def test_build_source_timeline_groups_mixed_dates_by_month_and_source():
    results = [
        {"source_project": "readwise", "created_at": "2024-02-20T10:30:00Z"},
        {
            "source_project": "max",
            "created_at": datetime(2024, 1, 5, 12, 0, tzinfo=timezone.utc),
        },
        {"source_project": "max", "created_at": date(2024, 1, 31)},
        {"source_project": "readwise", "created_at": "2024-01-01"},
    ]

    timeline = build_source_timeline(results)

    assert timeline["buckets"] == [
        {
            "bucket": "2024-01",
            "start": "2024-01-01",
            "sources": {"max": 2, "readwise": 1},
            "total": 3,
        },
        {
            "bucket": "2024-02",
            "start": "2024-02-01",
            "sources": {"readwise": 1},
            "total": 1,
        },
    ]
    assert timeline["sources"] == ["max", "readwise"]
    assert timeline["stats"]["included_count"] == 4
    assert timeline["stats"]["skipped_count"] == 0


def test_build_source_timeline_reports_missing_and_invalid_dates():
    timeline = build_source_timeline(
        [
            {"source_project": "max"},
            {"source_project": "readwise", "created_at": None},
            {"source_project": "csv", "created_at": "not-a-date"},
            {"source_project": "max", "created_at": 12345},
            {"source_project": "max", "created_at": "2024-03-01"},
        ]
    )

    assert timeline["buckets"] == [
        {
            "bucket": "2024-03",
            "start": "2024-03-01",
            "sources": {"max": 1},
            "total": 1,
        }
    ]
    assert timeline["stats"]["skipped_count"] == 4
    assert timeline["stats"]["skipped_missing_date"] == 2
    assert timeline["stats"]["skipped_invalid_date"] == 2


def test_build_source_timeline_reads_nested_unit_fields_and_unknown_source_fallback():
    timeline = build_source_timeline(
        [
            ResultStub(UnitStub(source_project="readwise", created_at="2024-04-15")),
            {
                "unit": {
                    "source_project": "nested",
                    "created_at": "2024-04-16T08:00:00+00:00",
                }
            },
            {
                "source_project": "flat",
                "created_at": "2024-04-17",
                "unit": {"source_project": "nested", "created_at": "2020-01-01"},
            },
            {"created_at": "2024-04-18"},
        ]
    )

    assert timeline["buckets"] == [
        {
            "bucket": "2024-04",
            "start": "2024-04-01",
            "sources": {"flat": 1, "nested": 1, "readwise": 1, "unknown": 1},
            "total": 4,
        }
    ]
    assert timeline["sources"] == ["flat", "nested", "readwise", "unknown"]


def test_build_source_timeline_supports_day_week_and_year_buckets():
    results = [
        {"source_project": "max", "created_at": "2024-01-01"},
        {"source_project": "max", "created_at": "2024-01-07"},
        {"source_project": "readwise", "created_at": "2024-01-08"},
    ]

    day_buckets = build_source_timeline(results, bucket="day")["buckets"]

    assert [item["bucket"] for item in day_buckets] == [
        "2024-01-01",
        "2024-01-07",
        "2024-01-08",
    ]
    assert build_source_timeline(results, bucket="week")["buckets"] == [
        {
            "bucket": "2024-W01",
            "start": "2024-01-01",
            "sources": {"max": 2},
            "total": 2,
        },
        {
            "bucket": "2024-W02",
            "start": "2024-01-08",
            "sources": {"readwise": 1},
            "total": 1,
        },
    ]
    assert build_source_timeline(results, bucket="year")["buckets"] == [
        {
            "bucket": "2024",
            "start": "2024-01-01",
            "sources": {"max": 2, "readwise": 1},
            "total": 3,
        }
    ]


def test_build_source_timeline_ordering_is_deterministic_across_input_order():
    results = [
        {"source_project": "readwise", "created_at": "2024-02-01"},
        {"source_project": "max", "created_at": "2024-01-01"},
        {"source_project": "csv", "created_at": "2024-01-15"},
    ]

    assert build_source_timeline(results)["buckets"] == build_source_timeline(
        reversed(results)
    )["buckets"]


def test_build_source_timeline_limit_applies_after_chronological_bucket_sorting():
    timeline = build_source_timeline(
        [
            {"source_project": "max", "created_at": "2024-03-01"},
            {"source_project": "max", "created_at": "2024-01-01"},
            {"source_project": "readwise", "created_at": "2024-02-01"},
        ],
        limit=2,
    )

    assert [bucket["bucket"] for bucket in timeline["buckets"]] == ["2024-01", "2024-02"]
    assert timeline["stats"]["candidate_count"] == 3
    assert timeline["stats"]["included_count"] == 2
    assert timeline["stats"]["omitted_buckets"] == 1
    assert timeline["stats"]["limit"] == 2


@pytest.mark.parametrize("bucket", ["hour", "", None])
def test_build_source_timeline_validates_bucket(bucket):
    with pytest.raises(ValueError, match="bucket must be one of: day, month, week, year"):
        build_source_timeline([], bucket=bucket)


@pytest.mark.parametrize("limit", [-1, 1.5, "2", True])
def test_build_source_timeline_validates_limit(limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        build_source_timeline([], limit=limit)


def test_build_source_timeline_is_importable_from_graph_rag():
    assert callable(build_source_timeline)
