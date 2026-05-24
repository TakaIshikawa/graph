from __future__ import annotations

from graph.rag.context_section_transition_plan import plan_context_section_transitions


def test_context_section_transition_plan_empty_input():
    assert plan_context_section_transitions([]) == {"sections": [], "transitions": []}


def test_context_section_transition_plan_uses_topic_source_date_and_conflict():
    records = [
        {"id": "a", "topic": "cost", "source_type": "study", "date": "2024-01-01", "text": "Cost evidence"},
        {"id": "b", "topic": "benefit", "source_type": "study", "date": "2024-02-01", "text": "Benefit evidence"},
        {"id": "c", "topic": "benefit", "source_type": "news", "date": "2025-01-01", "text": "However the benefit conflicts."},
    ]
    original = [dict(row) for row in records]

    result = plan_context_section_transitions(records)

    assert records == original
    assert [section["record_ids"] for section in result["sections"]] == [["a"], ["b"], ["c"]]
    assert result["transitions"][0]["reason_codes"] == ["topic_shift"]
    assert set(result["transitions"][1]["reason_codes"]) == {"source_shift", "date_shift", "conflict_cue"}
    assert result["transitions"][1]["label"] == "contrasting evidence"
