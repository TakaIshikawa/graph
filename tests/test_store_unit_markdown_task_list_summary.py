from __future__ import annotations

from graph.store.unit_markdown_task_list_summary import summarize_unit_markdown_task_lists


def test_task_list_summary_counts_checked_unchecked_and_ignores_fences():
    summary = summarize_unit_markdown_task_lists([
        {"id": "u2", "content": "- [ ] todo\n* [X] done\n```\n- [x] ignored\n```"},
        {"id": "u1", "content": "- [x] done"},
    ])

    assert summary == {
        "total_units": 2,
        "units_with_tasks": 2,
        "total_task_items": 3,
        "checked_task_count": 2,
        "unchecked_task_count": 1,
        "example_unit_ids": ["u1", "u2"],
        "low_count_unit_samples": [{"unit_id": "u1", "task_count": 1}, {"unit_id": "u2", "task_count": 2}],
    }
