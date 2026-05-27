from __future__ import annotations

from graph.store import summarize_unit_frontmatter_array_fields


def test_summarize_unit_frontmatter_array_fields_parses_leading_yaml_lists():
    summary = summarize_unit_frontmatter_array_fields([{"id": "a", "title": "A", "content": "---\ntags:\n  - one\n  - two\nrefs:\n  - name: obj\n  - scalar\n---\nBody"}, {"id": "b", "content": "tags:\n- nope"}])

    assert summary["array_fields"] == [
        {"key": "refs", "unit_count": 1, "min_items": 2, "max_items": 2, "total_items": 2, "mixed_type_count": 1, "sample_units": [{"unit_id": "a", "title": "A"}]},
        {"key": "tags", "unit_count": 1, "min_items": 2, "max_items": 2, "total_items": 2, "mixed_type_count": 0, "sample_units": [{"unit_id": "a", "title": "A"}]},
    ]
