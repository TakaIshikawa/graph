from graph.store import summarize_unit_markdown_heading_outlines


def test_heading_outline_summary_counts_atx_setext_skips_and_duplicates():
    content = "# Top\n### Skip\nTop\n---\n```\n# Hidden\n```"
    summary = summarize_unit_markdown_heading_outlines([{"id": "u1", "content": content}])

    assert summary["units_with_headings"] == 1
    assert summary["total_headings"] == 3
    assert summary["max_heading_depth"] == 3
    assert summary["skipped_level_issue_count"] == 1
    assert summary["duplicate_heading_text_count"] == 1
