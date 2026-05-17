from __future__ import annotations

import pytest

from graph.rag import allocate_context_token_budget


def test_context_token_budget_never_exceeds_total_budget():
    rows = allocate_context_token_budget(
        [{"id": "a", "score": 3}, {"id": "b", "score": 1}, {"id": "c", "score": 0}],
        total_budget=100,
        min_tokens_per_result=10,
    )

    assert sum(row["allocated_tokens"] for row in rows) == 100
    assert rows[0]["allocated_tokens"] >= rows[1]["allocated_tokens"] >= rows[2]["allocated_tokens"]


def test_context_token_budget_caps_minimum_when_budget_is_small():
    rows = allocate_context_token_budget([{"id": "a"}, {"id": "b"}, {"id": "c"}], total_budget=2, min_tokens_per_result=5)

    assert sum(row["allocated_tokens"] for row in rows) == 2
    assert [row["result_id"] for row in rows] == ["a", "b", "c"]


@pytest.mark.parametrize("total_budget", [-1, 1.5, True])
def test_context_token_budget_validates_total_budget(total_budget):
    with pytest.raises(ValueError, match="total_budget"):
        allocate_context_token_budget([], total_budget=total_budget)
