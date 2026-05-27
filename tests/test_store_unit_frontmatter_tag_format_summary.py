from graph.store import summarize_unit_frontmatter_tag_formats


def test_frontmatter_tag_formats_cover_scalar_list_empty_and_duplicates():
    report = summarize_unit_frontmatter_tag_formats(
        [
            {"id": "scalar", "metadata": {"tags": "alpha, beta"}},
            {"id": "list", "tags": ["alpha", "beta"]},
            {"id": "empty", "tags": []},
            {"id": "dup", "tags": ["Alpha", " alpha  "]},
        ]
    )

    assert report["issue_counts"]["scalar"] == 1
    assert report["issue_counts"]["list"] == 3
    assert report["issue_counts"]["empty"] == 1
    assert report["issue_counts"]["duplicate"] == 1
    assert report["issue_counts"]["whitespace"] == 1


def test_frontmatter_tag_formats_handles_content_frontmatter_and_non_strings():
    report = summarize_unit_frontmatter_tag_formats(
        [
            {"id": "fm", "content": "---\ntags: [one, two]\n---\nbody"},
            {"id": "num", "tags": [1, "1"]},
        ]
    )

    assert report["total_units"] == 2
    assert report["issue_counts"]["duplicate"] == 1
