from __future__ import annotations

from graph.store import summarize_unit_markdown_comment_directives


def test_comment_directive_summary_extracts_labels_from_html_comments_only():
    report = summarize_unit_markdown_comment_directives([{"id": "u", "content": "<!-- TODO: fix -->\nTODO outside\n<!-- plain words -->\n<!-- graph-key value -->"}])

    assert report["hidden_comment_blocks"] == 3
    assert report["plain_comment_blocks"] == 1
    assert report["units_with_directives"] == 1
    assert report["directive_labels"] == [{"label": "graph-key", "count": 1}, {"label": "todo", "count": 1}]
