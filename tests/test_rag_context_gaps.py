from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.rag import detect_context_gaps

NOW = datetime(2026, 5, 1, tzinfo=timezone.utc)


def test_detect_context_gaps_flags_source_diversity_gap():
    report = detect_context_gaps(
        [
            {
                "id": "unit-a",
                "source_project": "max",
                "domain": "Example.com",
                "content_type": "insight",
                "tags": ["solar"],
            },
            {
                "id": "unit-b",
                "source_project": "max",
                "url": "https://example.com/posts/b",
                "content_type": "finding",
                "tags": ["grid"],
            },
        ],
        min_sources=2,
        now=NOW,
    )

    assert report["coverage"]["source_projects"] == [
        {"value": "max", "count": 2}
    ]
    assert report["coverage"]["domains"] == [{"value": "example.com", "count": 2}]
    assert report["representative_result_ids"]["source_projects"] == {
        "max": ["unit-a", "unit-b"]
    }
    assert report["gaps"] == [
        {
            "type": "source_diversity",
            "severity": "warning",
            "message": "Retrieved context does not cover enough source projects.",
            "current": 1,
            "required": 2,
        }
    ]
    assert report["suggestions"] == [
        "Retrieve from at least 2 distinct source projects."
    ]


def test_detect_context_gaps_flags_missing_required_tags_and_facets():
    report = detect_context_gaps(
        [
            {
                "id": "unit-a",
                "source_project": "max",
                "source_url": "https://www.example.com/a",
                "content_type": "insight",
                "tags": ["solar"],
            }
        ],
        required_facets={
            "tags": ["solar", "storage"],
            "source_projects": ["max", "presence"],
            "domains": ["example.com", "missing.test"],
            "content_types": ["finding", "insight"],
        },
        min_sources=0,
        now=NOW,
    )

    assert report["coverage"]["tags"] == [{"value": "solar", "count": 1}]
    assert report["gaps"] == [
        {
            "type": "missing_required_tags",
            "severity": "warning",
            "message": "Required tags are missing.",
            "missing": ["storage"],
        },
        {
            "type": "missing_required_source_projects",
            "severity": "warning",
            "message": "Required source projects are missing.",
            "missing": ["presence"],
        },
        {
            "type": "missing_required_domains",
            "severity": "warning",
            "message": "Required domains are missing.",
            "missing": ["missing.test"],
        },
        {
            "type": "missing_required_content_types",
            "severity": "warning",
            "message": "Required content types are missing.",
            "missing": ["finding"],
        },
    ]
    assert report["suggestions"] == [
        "Add context matching tags: `storage`.",
        "Add context matching source_projects: `presence`.",
        "Add context matching domains: `missing.test`.",
        "Add context matching content_types: `finding`.",
    ]


def test_detect_context_gaps_flags_recency_requirement():
    report = detect_context_gaps(
        [
            {
                "id": "old",
                "source_project": "max",
                "updated_at": "2026-03-01T00:00:00Z",
            },
            {
                "id": "recent",
                "source_project": "presence",
                "metadata": {"published_at": "2026-04-20"},
            },
        ],
        min_recent_items=2,
        now=NOW,
    )

    assert report["coverage"]["recency"] == {
        "dated_count": 2,
        "recent_count": 1,
        "window_days": 30,
        "oldest_at": "2026-03-01T00:00:00+00:00",
        "newest_at": "2026-04-20T00:00:00+00:00",
    }
    assert report["representative_result_ids"]["recent"] == {
        "last_30_days": ["recent"]
    }
    assert report["gaps"] == [
        {
            "type": "recency",
            "severity": "warning",
            "message": "Retrieved context does not include enough recent items.",
            "current": 1,
            "required": 2,
            "window_days": 30,
        }
    ]


def test_detect_context_gaps_empty_results_returns_clear_gaps_and_suggestions():
    report = detect_context_gaps(
        [],
        required_facets={"tags": ["storage"]},
        min_sources=2,
        min_recent_items=1,
        now=NOW,
    )

    assert report["coverage"] == {
        "result_count": 0,
        "source_projects": [],
        "domains": [],
        "content_types": [],
        "tags": [],
        "recency": {
            "dated_count": 0,
            "recent_count": 0,
            "window_days": 30,
            "oldest_at": None,
            "newest_at": None,
        },
    }
    assert [gap["type"] for gap in report["gaps"]] == [
        "empty_results",
        "source_diversity",
        "missing_required_tags",
        "recency",
    ]
    assert report["suggestions"] == [
        "Run a broader retrieval before generating an answer.",
        "Retrieve from at least 2 distinct source projects.",
        "Add context matching tags: `storage`.",
        "Retrieve newer context from the last 30 days until at least 1 recent items are available.",
    ]


def test_detect_context_gaps_sorts_counts_suggestions_and_does_not_mutate():
    results = [
        {
            "id": "unit-c",
            "source_project": "beta",
            "content_type": "report",
            "tags": ["storage", "solar"],
        },
        {
            "id": "unit-a",
            "source_project": "alpha",
            "content_type": "insight",
            "tags": ["solar"],
        },
        {
            "id": "unit-b",
            "source_project": "alpha",
            "content_type": "report",
            "tags": ["wind"],
        },
    ]
    original_tags = list(results[0]["tags"])

    report = detect_context_gaps(
        results,
        required_facets={"tags": ["wind", "market"], "content_type": "brief"},
        min_sources=0,
        now=NOW,
    )

    assert results[0]["tags"] == original_tags
    assert report["coverage"]["source_projects"] == [
        {"value": "alpha", "count": 2},
        {"value": "beta", "count": 1},
    ]
    assert report["coverage"]["content_types"] == [
        {"value": "report", "count": 2},
        {"value": "insight", "count": 1},
    ]
    assert report["coverage"]["tags"] == [
        {"value": "solar", "count": 2},
        {"value": "storage", "count": 1},
        {"value": "wind", "count": 1},
    ]
    assert report["suggestions"] == [
        "Add context matching tags: `market`.",
        "Add context matching content_types: `brief`.",
    ]


def test_detect_context_gaps_is_importable_from_graph_rag():
    assert callable(detect_context_gaps)


@pytest.mark.parametrize("value", [-1, 1.5, "2", True])
def test_detect_context_gaps_rejects_invalid_min_sources(value):
    with pytest.raises(ValueError, match="min_sources must be a non-negative integer"):
        detect_context_gaps([], min_sources=value)


@pytest.mark.parametrize("value", [-1, 1.5, "2", True])
def test_detect_context_gaps_rejects_invalid_min_recent_items(value):
    with pytest.raises(
        ValueError,
        match="min_recent_items must be a non-negative integer",
    ):
        detect_context_gaps([], min_recent_items=value)
