from __future__ import annotations

from types import SimpleNamespace

from graph.rag.context_priority import plan_context_priority


def test_context_priority_orders_by_score_and_signals():
    plan = plan_context_priority(
        [
            {"id": "low", "score": 0.2, "content": "short"},
            {
                "id": "high",
                "score": 0.8,
                "content": "A" * 200,
                "url": "https://example.com",
                "source_id": "src",
                "published_at": "2026-04-01",
            },
        ]
    )

    assert [item["result_id"] for item in plan["items"]] == ["high", "low"]
    assert plan["items"][0]["rank"] == 1
    assert "citation present" in plan["items"][0]["reasons"]
    assert plan["counts"] == {"result_count": 2, "returned_count": 2, "source_count": 2}


def test_context_priority_handles_missing_metadata_and_max_items():
    plan = plan_context_priority(
        [
            {"id": "a", "content": "plain text"},
            {"id": "b", "content": "plain text", "score": 0.1},
        ],
        max_items=1,
    )

    assert len(plan["items"]) == 1
    assert plan["counts"]["result_count"] == 2
    assert plan["counts"]["returned_count"] == 1


def test_context_priority_stable_tie_breaking_and_objects():
    first = SimpleNamespace(id="first", score=0.5, text="A" * 150)
    second = SimpleNamespace(id="second", score=0.5, text="A" * 150)

    plan = plan_context_priority([first, second])

    assert [item["result_id"] for item in plan["items"]] == ["first", "second"]
    assert [item["rank"] for item in plan["items"]] == [1, 2]
