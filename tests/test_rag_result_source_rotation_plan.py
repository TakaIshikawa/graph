from __future__ import annotations

from graph.rag.result_source_rotation_plan import plan_result_source_rotation


def test_result_source_rotation_plan_empty_input():
    assert plan_result_source_rotation([]) == {"rotation": []}


def test_result_source_rotation_plan_reduces_adjacent_sources_preserving_front_score():
    result = plan_result_source_rotation(
        [
            {"id": "a", "source_id": "s1", "score": 0.99},
            {"id": "b", "source_id": "s1", "score": 0.98},
            {"id": "c", "source_id": "s2", "score": 0.8},
            {"id": "d", "source_id": "s1", "score": 0.7},
        ]
    )

    rotation = result["rotation"]
    assert [row["result_id"] for row in rotation] == ["a", "c", "b", "d"]
    assert rotation[0]["movement_reason"] == "preserve_score_order"
    assert rotation[1]["movement_reason"] == "source_diversity"
    assert rotation[0]["score"] == 0.99
