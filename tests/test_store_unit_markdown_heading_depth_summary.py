from graph.store import summarize_unit_markdown_heading_depths


def test_heading_depth_summary_counts_skips_and_ignores_fences():
    result = summarize_unit_markdown_heading_depths(
        [{"id": "u", "content": "# One\n## Two\n#### Four\n```\n# Hidden\n```\n### Three"}],
        sample_limit=2,
    )

    assert result["total_units"] == 1
    assert result["units_with_headings"] == 1
    assert result["heading_count"] == 4
    assert result["depth_counts"]["1"] == 1
    assert result["depth_counts"]["4"] == 1
    assert result["max_depth"] == 4
    assert result["skipped_level_count"] == 1
    assert len(result["samples"]) == 2
