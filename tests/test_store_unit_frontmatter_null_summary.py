from graph.store import summarize_unit_frontmatter_nulls


def test_summary_detects_frontmatter_null_kinds_by_key():
    summary = summarize_unit_frontmatter_nulls([{"content": "---\na: null\nb: ~\nc:\n---\na: null"}])
    assert summary["total_null_values"] == 3
    assert summary["units_with_null_values"] == 1
    assert summary["key_counts"] == {"a": 1, "b": 1, "c": 1}
    assert summary["null_kind_counts"] == {"blank": 1, "null": 1, "tilde": 1}


def test_only_frontmatter_is_scanned():
    assert summarize_unit_frontmatter_nulls([{"content": "a: null"}])["total_null_values"] == 0
