from __future__ import annotations

import pytest

from graph.rag.context_window_packing import plan_context_window_packing


def test_context_window_packing_never_exceeds_available_budget_and_preserves_order():
    report = plan_context_window_packing(
        [
            {"id": "a", "snippet": "one two three"},
            {"id": "b", "snippet": "four five six seven"},
            {"id": "c", "snippet": "eight nine"},
        ],
        token_budget=8,
        reserved_tokens=1,
    )

    assert report["available_tokens"] == 7
    assert report["used_tokens"] == 7
    assert [row["result_id"] for row in report["selected"]] == ["a", "b"]
    assert report["selected"][0]["selection"] == "full"
    assert report["selected"][1]["selection"] == "full"
    assert report["omitted"][0]["result_id"] == "c"


def test_context_window_packing_marks_truncated_selection():
    report = plan_context_window_packing([{"id": "long", "text": "alpha beta gamma delta"}], token_budget=3)

    assert report["selected"] == [
        {
            "result_id": "long",
            "title": None,
            "estimated_tokens": 4,
            "used_tokens": 3,
            "selection": "truncated",
            "text": "alpha beta gamma",
        }
    ]
    assert report["warnings"] == ["some_results_truncated"]


def test_context_window_packing_validates_budgets():
    with pytest.raises(ValueError):
        plan_context_window_packing([], token_budget=-1)
    with pytest.raises(ValueError):
        plan_context_window_packing([], token_budget=1, reserved_tokens=-1)
