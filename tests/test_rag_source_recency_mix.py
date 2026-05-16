from __future__ import annotations

from types import SimpleNamespace

from graph.rag.source_recency_mix import analyze_source_recency_mix


def test_source_recency_mix_buckets_with_fixed_now():
    mix = analyze_source_recency_mix(
        [
            {"id": "recent", "source_id": "s1", "published_at": "2026-04-20"},
            {"id": "current", "source_id": "s1", "updated_at": "2025-10-01"},
            {"id": "aging", "source_id": "s2", "created_at": "2024-01-01"},
            {"id": "stale", "source_id": "s2", "date": "2020-01-01"},
            {"id": "bad", "source_id": "s3", "date": "not-a-date"},
        ],
        now="2026-05-01",
    )

    assert mix["overall"] == {
        "recent": 1,
        "current": 1,
        "aging": 1,
        "stale": 1,
        "undated": 1,
    }
    assert mix["oldest_date"] == "2020-01-01"
    assert mix["newest_date"] == "2026-04-20"


def test_source_recency_mix_groups_per_source_and_objects():
    mix = analyze_source_recency_mix(
        [
            SimpleNamespace(id="a", source_id="docs", published_at="2026-04-01"),
            SimpleNamespace(id="b", source_id="docs", published_at=None),
        ],
        now="2026-04-15",
    )

    assert mix["sources"] == [
        {
            "source": "docs",
            "total": 2,
            "buckets": {
                "recent": 1,
                "current": 0,
                "aging": 0,
                "stale": 0,
                "undated": 1,
            },
        }
    ]


def test_source_recency_mix_empty_input():
    assert analyze_source_recency_mix([], now="2026-01-01") == {
        "total_results": 0,
        "as_of": "2026-01-01",
        "overall": {
            "recent": 0,
            "current": 0,
            "aging": 0,
            "stale": 0,
            "undated": 0,
        },
        "sources": [],
        "oldest_date": None,
        "newest_date": None,
    }
