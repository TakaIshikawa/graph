from __future__ import annotations

from graph.store import summarize_unit_frontmatter_tag_cardinality


def test_frontmatter_tag_cardinality_summary_parses_scalar_inline_and_block_tags():
    summary = summarize_unit_frontmatter_tag_cardinality(
        [
            {"id": "u1", "content": "---\ntags: Alpha, beta\n---"},
            {"id": "u2", "content": "---\ntags: [beta, Gamma, beta]\n---"},
            {"id": "u3", "content": "---\ntags:\n  - delta\n  - epsilon\n---"},
            {"id": "u4", "content": "---\ntitle: None\n---"},
        ],
        high_cardinality_threshold=2,
    )

    assert summary["distinct_normalized_tag_count"] == 5
    assert summary["units_with_no_tags"] == ["u4"]
    assert summary["duplicate_tag_units"] == [{"unit_id": "u2", "duplicate_tags": ["beta"]}]
    assert summary["high_cardinality_units"] == [{"unit_id": "u1", "tag_count": 2}, {"unit_id": "u2", "tag_count": 2}, {"unit_id": "u3", "tag_count": 2}]
    assert summary["top_tags"][0] == {"tag": "beta", "unit_count": 3}
