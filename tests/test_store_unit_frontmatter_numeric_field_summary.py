from graph.store import summarize_unit_frontmatter_numeric_fields


def test_frontmatter_numeric_summary_counts_nested_ints_floats_and_negatives():
    summary = summarize_unit_frontmatter_numeric_fields(
        [{"id": "u", "content": "---\ncount: 3\nratio: 0.5\nflag: true\nnested:\n  debt: -2\n---\nBody"}]
    )
    assert summary["total_units"] == 1
    assert summary["units_with_numeric_frontmatter"] == 1
    assert summary["field_counts"] == {"count": 1, "nested.debt": 1, "ratio": 1}
    assert summary["type_counts"] == {"float": 1, "integer": 2}
    assert summary["negative_value_counts"] == {"nested.debt": 1}


def test_frontmatter_numeric_summary_skips_invalid_frontmatter():
    assert summarize_unit_frontmatter_numeric_fields([{"content": "---\ncount: 3"}])["field_counts"] == {}
