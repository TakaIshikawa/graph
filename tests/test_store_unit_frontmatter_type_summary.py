from graph.store import summarize_unit_frontmatter_types


def test_unit_frontmatter_types_groups_metadata_and_yaml_value_types():
    report = summarize_unit_frontmatter_types(
        [
            {"id": "a", "metadata": {"frontmatter": {"title": "Doc", "tags": ["a"], "draft": False, "count": 3, "config": {"x": 1}, "empty": None}}},
            {"id": "b", "content": "---\ntitle: 2025-04-30\ntags: one\ndraft: true\ncount: 4.5\n---\nBody"},
            {"id": "c", "content": "---\ntags:\n  - one\n---\nBody"},
        ]
    )

    assert report["unit_count"] == 3
    assert report["frontmatter_unit_count"] == 3
    assert report["type_counts_by_key"]["title"] == [{"type": "date-like string", "count": 1}, {"type": "scalar", "count": 1}]
    assert report["type_counts_by_key"]["tags"] == [{"type": "list", "count": 2}, {"type": "scalar", "count": 1}]
    assert report["type_counts_by_key"]["draft"] == [{"type": "boolean", "count": 2}]
    assert report["type_counts_by_key"]["count"] == [{"type": "number", "count": 2}]
    assert report["type_counts_by_key"]["config"] == [{"type": "dict", "count": 1}]
    assert report["type_counts_by_key"]["empty"] == [{"type": "null", "count": 1}]
    assert report["mixed_type_keys"] == ["tags", "title"]


def test_unit_frontmatter_types_zero_safe():
    assert summarize_unit_frontmatter_types([{"content": "Body"}]) == {
        "unit_count": 1,
        "frontmatter_unit_count": 0,
        "type_counts_by_key": {},
        "mixed_type_keys": [],
        "samples": {},
    }
