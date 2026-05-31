from __future__ import annotations

from graph.store.unit_markdown_task_tag_summary import summarize_unit_markdown_task_tags


def test_task_tag_summary_only_analyzes_task_lines_and_tracks_state():
    summary = summarize_unit_markdown_task_tags(
        [
            {"id": "u1", "content": "- [ ] call #waiting @home\n- [x] ship #waiting +launch\n#waiting not a task"},
            {"id": "u2", "content": "```\n- [ ] fake #blocked\n```\n- [ ] real #blocked"},
        ]
    )

    by_tag = {row["marker"]: row for row in summary["tags"]}
    assert summary["task_count"] == 3
    assert by_tag["#waiting"]["task_count"] == 2
    assert by_tag["#waiting"]["checked_count"] == 1
    assert by_tag["#waiting"]["unchecked_count"] == 1
    assert by_tag["#blocked"]["task_count"] == 1
