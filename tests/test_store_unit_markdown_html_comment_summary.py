from __future__ import annotations

from graph.store import summarize_unit_markdown_html_comments


def test_html_comment_summary_counts_single_multiline_and_todo_like_comments():
    summary = summarize_unit_markdown_html_comments(
        [
            {"id": "u1", "content": "A <!-- TODO: fix --> B\n<!-- long\nNOTE text -->"},
            {"id": "u2", "content": "No comments"},
        ]
    )

    assert summary["affected_unit_count"] == 1
    assert summary["total_comment_count"] == 2
    assert summary["multiline_comment_count"] == 1
    assert summary["todo_like_comment_count"] == 2
    assert summary["units"][0]["longest_comment_samples"] == ["long NOTE text", "TODO: fix"]
