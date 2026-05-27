from graph.store import summarize_unit_markdown_details


def test_details_summary_counts_open_missing_summary_and_unclosed():
    content = "<details open><summary>One</summary>\n</details>\n<details>\nbody"
    summary = summarize_unit_markdown_details([{"id": "u", "content": content}])
    assert summary["total_units"] == 1
    assert summary["units_with_details"] == 1
    assert summary["details_count"] == 2
    assert summary["open_details_count"] == 1
    assert summary["missing_summary_count"] == 1
    assert summary["unclosed_details_count"] == 1
    assert summary["samples"][0] == {"unit_id": "u", "start_line": 1, "summary": "One", "is_open": True}
