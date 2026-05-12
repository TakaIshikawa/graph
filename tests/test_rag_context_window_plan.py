from __future__ import annotations

import pytest

from graph.rag.context_window_plan import plan_context_window


def test_context_window_plan_includes_within_budget_and_reports_overflow():
    plan = plan_context_window(
        [
            {"id": "a", "title": "alpha beta", "score": 0.9, "source_project": "one"},
            {"id": "b", "content": "gamma delta epsilon", "score": 0.8, "source_project": "one"},
            ({"id": "c", "summary": "zeta eta", "source_project": "two"}, 0.7),
        ],
        token_budget=6,
        reserve_tokens=1,
    )

    assert [row["id"] for row in plan["included"]] == ["a", "b"]
    assert [row["id"] for row in plan["excluded"]] == ["c"]
    assert plan["used_tokens"] == 5
    assert plan["available_tokens"] == 5
    assert plan["overflow_count"] == 1


def test_context_window_plan_honors_min_per_source_when_possible():
    plan = plan_context_window(
        [
            {"id": "a", "title": "one two three", "score": 1, "source_project": "a"},
            {"id": "b", "title": "one", "score": 0.2, "source_project": "b"},
            {"id": "c", "title": "one", "score": 0.9, "source_project": "a"},
        ],
        token_budget=4,
        min_per_source=1,
    )

    assert [row["id"] for row in plan["included"]] == ["a", "b"]


@pytest.mark.parametrize("kwargs", [{"token_budget": -1}, {"token_budget": 1, "reserve_tokens": True}, {"token_budget": 1, "min_per_source": 1.2}])
def test_context_window_plan_validates_integer_options(kwargs):
    with pytest.raises(ValueError):
        plan_context_window([], **kwargs)
