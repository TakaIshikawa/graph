from graph.store.unit_markdown_heading_numbering_summary import summarize_unit_markdown_heading_numbering


def test_summarizes_numbered_headings_repeats_and_skips():
    report = summarize_unit_markdown_heading_numbering([
        {"id": "b", "content": "# 1. Start\n## 1.2 Detail\n# 1. Repeat\n```\n# 2 Hidden\n```"},
        {"id": "a", "content": "# Intro\n# 01 Preface\n# 3. Later\n# 4 Next"},
    ])

    assert report["total_headings"] == 7
    assert report["numbered_headings"] == 6
    assert report["units_with_numbered_headings"] == 2
    assert report["numbering_depth_counts"] == {"1": 5, "2": 1}
    assert report["repeated_number_samples"] == [{"unit_id": "b", "line_number": 3, "number": "1", "heading": "1. Repeat"}]
    assert report["skipped_sequence_samples"] == [{"unit_id": "a", "line_number": 3, "expected": "2", "actual": "3", "heading": "3. Later"}]
