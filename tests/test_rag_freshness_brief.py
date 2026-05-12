from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graph.rag.freshness_brief import build_freshness_brief
from graph.types.models import KnowledgeUnit


NOW = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)


def test_freshness_brief_buckets_dict_inputs_and_metadata_timestamps():
    brief = build_freshness_brief(
        [
            {
                "id": "fresh-1",
                "title": "Fresh",
                "source_project": "notes",
                "updated_at": "2026-05-10T12:00:00Z",
            },
            {
                "id": "aging-1",
                "title": "Aging",
                "metadata": {
                    "source_project": "books",
                    "published_at": "2026-03-01",
                },
            },
            {
                "id": "stale-1",
                "title": "Stale",
                "source_project": "books",
                "created_at": "2025-01-01T00:00:00Z",
            },
            {"id": "undated-1", "title": "Undated", "source_project": "notes"},
        ],
        now=NOW,
        fresh_window_days=30,
        stale_window_days=180,
    )

    assert brief["counts"] == {"fresh": 1, "aging": 1, "stale": 1, "undated": 1}
    assert brief["buckets"]["fresh"]["source_distribution"] == {"notes": 1}
    assert brief["buckets"]["aging"]["source_distribution"] == {"books": 1}
    assert brief["buckets"]["stale"]["source_distribution"] == {"books": 1}
    assert brief["buckets"]["undated"]["source_distribution"] == {"notes": 1}
    assert brief["buckets"]["aging"]["results"] == [
        {
            "id": "aging-1",
            "title": "Aging",
            "source_project": "books",
            "timestamp": "2026-03-01T00:00:00+00:00",
            "bucket": "aging",
        }
    ]


def test_freshness_brief_accepts_model_like_objects_and_nested_units():
    result = SimpleNamespace(
        id="wrapper",
        source_project="wrapper-source",
        unit=KnowledgeUnit(
            id="unit-1",
            source_project="kindle",
            source_id="k1",
            source_entity_type="highlight",
            title="Nested Unit",
            content="Text",
            metadata={"updated_at": "2026-05-11T12:00:00Z"},
        ),
    )

    brief = build_freshness_brief([result], now=NOW)

    assert brief["counts"]["fresh"] == 1
    assert brief["buckets"]["fresh"]["results"][0]["id"] == "wrapper"
    assert brief["buckets"]["fresh"]["results"][0]["title"] == "Nested Unit"
    assert brief["buckets"]["fresh"]["results"][0]["source_project"] == "wrapper-source"


def test_freshness_brief_accepts_unit_score_tuples():
    unit = KnowledgeUnit(
        id="unit-1",
        source_project="readwise",
        source_id="r1",
        source_entity_type="highlight",
        title="Scored Unit",
        content="Text",
        updated_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    brief = build_freshness_brief([(unit, 0.82)], now=NOW)

    assert brief["buckets"]["fresh"]["results"] == [
        {
            "id": "unit-1",
            "title": "Scored Unit",
            "source_project": "readwise",
            "timestamp": "2026-05-01T00:00:00+00:00",
            "bucket": "fresh",
            "score": 0.82,
        }
    ]


def test_freshness_brief_normalizes_naive_and_aware_datetimes_deterministically():
    brief = build_freshness_brief(
        [
            {
                "id": "aware",
                "title": "Aware",
                "source_project": "calendar",
                "updated_at": "2026-05-12T21:00:00+09:00",
            },
            {
                "id": "naive",
                "title": "Naive",
                "source_project": "calendar",
                "updated_at": datetime(2026, 5, 12, 12, 0),
            },
        ],
        now=datetime(2026, 5, 12, 12, 0),
        limit=5,
    )

    assert brief["now"] == "2026-05-12T12:00:00+00:00"
    assert [row["id"] for row in brief["buckets"]["fresh"]["results"]] == [
        "aware",
        "naive",
    ]
    assert {row["timestamp"] for row in brief["buckets"]["fresh"]["results"]} == {
        "2026-05-12T12:00:00+00:00"
    }


def test_freshness_brief_caps_representatives_and_orders_by_recency_then_identity():
    brief = build_freshness_brief(
        [
            {
                "id": "b",
                "title": "B",
                "source_project": "z",
                "updated_at": "2026-05-10T00:00:00Z",
            },
            {
                "id": "a",
                "title": "A",
                "source_project": "a",
                "updated_at": "2026-05-11T00:00:00Z",
            },
            {
                "id": "c",
                "title": "C",
                "source_project": "a",
                "updated_at": "2026-05-10T00:00:00Z",
            },
        ],
        now=NOW,
        limit=2,
    )

    assert brief["counts"]["fresh"] == 3
    assert [row["id"] for row in brief["buckets"]["fresh"]["results"]] == ["a", "c"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fresh_window_days": 0}, "fresh_window_days must be a positive integer"),
        ({"fresh_window_days": True}, "fresh_window_days must be a positive integer"),
        ({"stale_window_days": 30}, "stale_window_days must be greater"),
        ({"stale_window_days": 1.5}, "stale_window_days must be a positive integer"),
        ({"limit": -1}, "limit must be a non-negative integer"),
        ({"limit": True}, "limit must be a non-negative integer"),
    ],
)
def test_freshness_brief_validates_windows_and_limit(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_freshness_brief([], now=NOW, **kwargs)
