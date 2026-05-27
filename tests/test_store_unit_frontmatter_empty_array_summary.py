from graph.store import summarize_unit_frontmatter_empty_arrays


def test_summary_detects_inline_and_empty_block_arrays():
    summary = summarize_unit_frontmatter_empty_arrays([{"content": "---\ntags: []\nempty:\nnext: value\n---"}])
    assert summary["total_empty_arrays"] == 2
    assert summary["units_with_empty_arrays"] == 1
    assert summary["key_counts"] == {"empty": 1, "tags": 1}
    assert summary["syntax_counts"] == {"block": 1, "inline": 1}


def test_non_empty_arrays_and_malformed_frontmatter_are_ignored():
    summary = summarize_unit_frontmatter_empty_arrays([{"content": "---\ntags:\n  - one\n---"}, {"content": "---\ntags: []"}])
    assert summary["total_empty_arrays"] == 0
