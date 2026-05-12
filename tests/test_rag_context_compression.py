from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag.context_compression import plan_context_compression


@dataclass
class ResultStub:
    id: str
    content: str
    metadata: dict


def by_id(rows: list[dict], result_id: str) -> dict:
    return next(row for row in rows if row["result_id"] == result_id)


def test_plan_context_compression_allocates_more_budget_to_higher_priority_results():
    payload = plan_context_compression(
        [
            {
                "id": "low",
                "content": " ".join(f"low{i}" for i in range(30)),
                "confidence": 0.2,
            },
            {
                "id": "high",
                "content": " ".join(f"high{i}" for i in range(30)),
                "confidence": 0.9,
            },
        ],
        token_budget=40,
    )

    assert payload["allocated_tokens"] <= 40
    assert by_id(payload["allocations"], "high")["allocated_tokens"] > by_id(
        payload["allocations"], "low"
    )["allocated_tokens"]
    assert [row["result_id"] for row in payload["allocations"]] == ["high", "low"]


def test_plan_context_compression_never_exceeds_budget_and_reports_dropped_ids():
    payload = plan_context_compression(
        [
            {"id": "a", "content": " ".join("a" for _ in range(20)), "score": 0.8},
            {"id": "b", "content": " ".join("b" for _ in range(20)), "score": 0.7},
            {"id": "empty", "content": "", "score": 1.0},
        ],
        token_budget=18,
    )

    assert payload["allocated_tokens"] == 15
    assert payload["remaining_tokens"] == 3
    assert payload["dropped_result_ids"] == ["empty", "b"]
    assert payload["dropped"] == [
        {"result_id": "empty", "reason": "no text to include"},
        {"result_id": "b", "reason": "insufficient remaining budget"},
    ]


def test_plan_context_compression_handles_empty_results_and_tiny_budgets():
    assert plan_context_compression([], token_budget=5) == {
        "token_budget": 5,
        "allocated_tokens": 0,
        "remaining_tokens": 5,
        "allocations": [],
        "dropped_result_ids": [],
        "dropped": [],
    }

    tiny = plan_context_compression(
        [
            {"id": "first", "snippet": "one two three", "confidence": 0.9},
            {"id": "second", "snippet": "four five six", "confidence": 0.8},
        ],
        token_budget=1,
    )
    assert tiny["allocated_tokens"] == 1
    assert tiny["allocations"] == [
        {
            "result_id": "first",
            "allocated_tokens": 1,
            "estimated_tokens": 3,
            "priority": 0.9,
            "action": "trim",
        }
    ]
    assert tiny["dropped_result_ids"] == ["second"]


def test_plan_context_compression_supports_objects_and_nested_metadata():
    payload = plan_context_compression(
        [
            ResultStub(
                id="object",
                content="alpha beta gamma delta",
                metadata={"confidence": "80"},
            ),
            {
                "unit": {
                    "id": "nested",
                    "content": "epsilon zeta eta theta iota",
                    "metadata": {"score": 0.5},
                }
            },
        ],
        token_budget=20,
    )

    assert payload["allocations"] == [
        {
            "result_id": "object",
            "allocated_tokens": 4,
            "estimated_tokens": 4,
            "priority": 0.8,
            "action": "include",
        },
        {
            "result_id": "nested",
            "allocated_tokens": 5,
            "estimated_tokens": 5,
            "priority": 0.5,
            "action": "include",
        },
    ]


@pytest.mark.parametrize("token_budget", [-1, 1.5, "10", True])
def test_plan_context_compression_validates_token_budget(token_budget):
    with pytest.raises(ValueError, match="token_budget must be a non-negative integer"):
        plan_context_compression([], token_budget=token_budget)
