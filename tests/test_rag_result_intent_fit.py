from __future__ import annotations

from graph.rag.result_intent_fit import score_result_intent_fit


def test_result_intent_fit_rewards_comparison_language():
    rows = score_result_intent_fit(
        "Compare Postgres vs SQLite performance",
        [
            {"id": "a", "content": "Postgres vs SQLite benchmark comparison with 42 tests", "source": "lab"},
            {"id": "b", "content": "SQLite notes only"},
        ],
    )

    assert rows[0]["intent"] == "comparison"
    assert rows[0]["fit_score"] > rows[1]["fit_score"]
    assert "comparison_language" in rows[0]["matched_signals"]
    assert all(0.0 <= row["fit_score"] <= 1.0 for row in rows)


def test_result_intent_fit_rewards_timeline_dates_and_handles_missing_content():
    rows = score_result_intent_fit(
        "timeline of model releases",
        [
            {"id": "dated", "title": "Model release timeline", "published_at": "2025-03-01"},
            {"id": "empty"},
        ],
    )

    assert rows[0]["intent"] == "timeline"
    assert "date_metadata" in rows[0]["matched_signals"]
    assert rows[1]["warnings"] == ["missing_content", "weak_intent_fit"]
