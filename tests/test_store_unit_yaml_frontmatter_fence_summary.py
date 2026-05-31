from graph.store import summarize_unit_yaml_frontmatter_fences


def test_frontmatter_fence_summary_counts_leading_closed_and_unclosed_fences():
    result = summarize_unit_yaml_frontmatter_fences(
        [
            {"id": "a", "content": "---\ntitle: A\n---\nBody\n---"},
            {"id": "b", "content": "+++\ntitle = 'B'"},
            {"id": "c", "content": "Body\n---\nnot frontmatter"},
        ]
    )

    assert result["total_units"] == 3
    assert result["units_with_frontmatter_fence"] == 2
    assert result["closed_fence_count"] == 1
    assert result["unclosed_fence_count"] == 1
    assert result["yaml_fence_count"] == 1
    assert result["toml_fence_count"] == 1
    assert result["samples"][1]["is_closed"] is False
