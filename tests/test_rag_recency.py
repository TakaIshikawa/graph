from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.rag import rerank_for_recency

NOW = datetime(2026, 5, 1, tzinfo=timezone.utc)


def by_id(results: list[dict], result_id: str) -> dict:
    return next(item for item in results if item["id"] == result_id)


def test_recent_dated_results_receive_higher_recency_score_than_older_results():
    results = rerank_for_recency(
        [
            {"id": "old", "score": 0.5, "created_at": "2025-05-01"},
            {"id": "recent", "score": 0.5, "created_at": "2026-04-15"},
        ],
        now=NOW,
    )

    assert by_id(results, "recent")["recency_score"] > by_id(results, "old")[
        "recency_score"
    ]
    assert [item["id"] for item in results] == ["recent", "old"]


def test_combined_score_respects_score_weight_and_existing_score_fields():
    results = rerank_for_recency(
        [
            {"id": "fresh", "score": 0.4, "created_at": "2026-05-01"},
            {"id": "relevant", "score": 1.0, "created_at": "2025-05-01"},
            {
                "id": "explained",
                "explanation": {"scores": {"final_score": 0.8}},
                "created_at": "2026-05-01",
            },
        ],
        score_weight=0.75,
        now=NOW,
    )

    assert by_id(results, "fresh")["combined_score"] == pytest.approx(0.55)
    assert by_id(results, "relevant")["combined_score"] > by_id(results, "fresh")[
        "combined_score"
    ]
    assert by_id(results, "explained")["combined_score"] == pytest.approx(0.85)
    assert [item["id"] for item in results] == ["explained", "relevant", "fresh"]


def test_missing_and_invalid_dates_do_not_raise_and_sort_below_dated_relevance_ties():
    source = [
        {"id": "missing", "score": 0.8},
        {"id": "invalid", "score": 0.8, "created_at": "not-a-date"},
        {"id": "dated", "score": 0.8, "created_at": "2026-01-01"},
    ]

    results = rerank_for_recency(source, now=NOW)

    assert [item["id"] for item in results] == ["dated", "invalid", "missing"]
    assert by_id(results, "missing")["recency_score"] == 0.0
    assert by_id(results, "invalid")["recency_score"] == 0.0
    assert "recency_score" not in source[0]


def test_uses_the_most_recent_valid_configured_date_key():
    results = rerank_for_recency(
        [
            {
                "id": "created-only",
                "score": 0.5,
                "created_at": "2026-04-01",
            },
            {
                "id": "updated",
                "score": 0.5,
                "created_at": "2025-01-01",
                "updated_at": "2026-04-15T12:00:00Z",
            },
        ],
        now=NOW,
    )

    assert [item["id"] for item in results] == ["updated", "created-only"]


def test_limit_applies_after_reranking_and_accepts_zero():
    results = [
        {"id": "old", "score": 0.5, "created_at": "2025-01-01"},
        {"id": "new", "score": 0.5, "created_at": "2026-05-01"},
    ]

    assert [item["id"] for item in rerank_for_recency(results, now=NOW, limit=1)] == [
        "new"
    ]
    assert rerank_for_recency(results, now=NOW, limit=0) == []


def test_rerank_for_recency_is_deterministic_across_reversed_equivalent_inputs():
    results = [
        {"id": "c", "score": 0.5},
        {"id": "a", "score": 0.5},
        {"id": "b", "score": 0.5},
    ]

    first = rerank_for_recency(results, now=NOW)
    second = rerank_for_recency(reversed(results), now=NOW)

    assert first == second
    assert [item["id"] for item in first] == ["a", "b", "c"]


@pytest.mark.parametrize("half_life_days", [0, -1, 1.5, "90", True])
def test_rerank_for_recency_validates_half_life_days(half_life_days):
    with pytest.raises(ValueError, match="half_life_days must be a positive integer"):
        rerank_for_recency([], half_life_days=half_life_days)


@pytest.mark.parametrize("score_weight", [-0.1, 1.1, "0.75", True])
def test_rerank_for_recency_validates_score_weight(score_weight):
    with pytest.raises(ValueError, match="score_weight must be"):
        rerank_for_recency([], score_weight=score_weight)


@pytest.mark.parametrize("limit", [-1, 1.5, "2", True])
def test_rerank_for_recency_validates_limit(limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer or None"):
        rerank_for_recency([], limit=limit)


@pytest.mark.parametrize("date_keys", [(), "", ("",), (None,)])
def test_rerank_for_recency_validates_date_keys(date_keys):
    with pytest.raises(
        ValueError, match="date_keys must be a non-empty sequence of strings"
    ):
        rerank_for_recency([], date_keys=date_keys)


def test_rerank_for_recency_is_importable_from_graph_rag():
    assert callable(rerank_for_recency)
