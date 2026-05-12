from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

import pytest

from graph.rag.result_freshness import score_result_freshness


@dataclass
class ResultStub:
    id: str
    metadata: dict


def by_id(rows: list[dict], result_id: str) -> dict:
    return next(row for row in rows if row["result_id"] == result_id)


def test_score_result_freshness_scores_recent_results_higher_than_old_results():
    rows = score_result_freshness(
        [
            {"id": "old", "published_at": "2024-05-13"},
            {"id": "recent", "published_at": "2026-05-12"},
        ],
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        half_life_days=365,
    )

    assert by_id(rows, "recent")["freshness_score"] > by_id(rows, "old")["freshness_score"]
    assert by_id(rows, "recent") == {
        "result_id": "recent",
        "freshness_score": 0.998103,
        "age_days": 1.0,
        "reason": "freshness from published_at",
    }


def test_score_result_freshness_handles_missing_invalid_and_fallback_ids():
    rows = score_result_freshness(
        [
            {"id": "missing"},
            {"id": "invalid", "metadata": {"updated_at": "not-a-date"}},
            {"title": "No explicit id", "published_at": ""},
        ],
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )

    assert rows == [
        {
            "result_id": "missing",
            "freshness_score": 0.0,
            "age_days": None,
            "reason": "missing date metadata",
        },
        {
            "result_id": "invalid",
            "freshness_score": 0.0,
            "age_days": None,
            "reason": "invalid date metadata",
        },
        {
            "result_id": "result-3",
            "freshness_score": 0.0,
            "age_days": None,
            "reason": "missing date metadata",
        },
    ]


def test_score_result_freshness_parses_timezone_aware_and_nested_dates():
    rows = score_result_freshness(
        [
            {
                "unit": {
                    "id": "nested",
                    "metadata": {"published_at": "2026-05-12T20:00:00+09:00"},
                }
            },
            ResultStub(id="object", metadata={"created_at": date(2026, 5, 10)}),
            ({"id": "tuple", "source": {"updated_at": "2026-05-13T01:00:00+01:00"}}, 0.9),
        ],
        now=datetime(2026, 5, 13, 0, 0, tzinfo=timezone.utc),
        half_life_days=10,
    )

    assert by_id(rows, "nested")["age_days"] == pytest.approx(13 / 24)
    assert by_id(rows, "object")["age_days"] == 3.0
    assert by_id(rows, "tuple") == {
        "result_id": "tuple",
        "freshness_score": 1.0,
        "age_days": 0.0,
        "reason": "freshness from updated_at",
    }


def test_score_result_freshness_treats_future_dates_as_current():
    rows = score_result_freshness(
        [{"id": "future", "published_at": "2026-05-14T00:00:00Z"}],
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )

    assert rows == [
        {
            "result_id": "future",
            "freshness_score": 1.0,
            "age_days": 0.0,
            "reason": "future published_at treated as current",
        }
    ]


@pytest.mark.parametrize("half_life_days", [0, -1, "365", True])
def test_score_result_freshness_validates_half_life(half_life_days):
    with pytest.raises(ValueError, match="half_life_days must be a positive number"):
        score_result_freshness([], half_life_days=half_life_days)
