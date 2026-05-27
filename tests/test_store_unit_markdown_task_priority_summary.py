from __future__ import annotations

from graph.store.unit_markdown_task_priority_summary import summarize_unit_markdown_task_priorities


def test_summarize_unit_markdown_task_priorities_counts_by_priority():
    summary = summarize_unit_markdown_task_priorities(
        [
            {"id": "u1", "content": "- [ ] #priority/high Ship\n- [x] priority:: high Done\n```md\n- [ ] #priority/low Ignore\n```"},
            {"id": "u2", "content": "- [ ] !! Review"},
        ],
        sample_limit=1,
    )

    assert summary["total_units"] == 2
    assert summary["priorities"] == [
        {"priority": "high", "task_count": 2, "unit_count": 1, "checked_count": 1, "unchecked_count": 1, "examples": [{"unit_id": "u1", "line": 1, "task_text": "Ship"}]},
        {"priority": "medium", "task_count": 1, "unit_count": 1, "checked_count": 0, "unchecked_count": 1, "examples": [{"unit_id": "u2", "line": 1, "task_text": "Review"}]},
    ]
