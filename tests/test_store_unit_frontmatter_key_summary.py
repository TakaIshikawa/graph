from __future__ import annotations

from graph.store.unit_frontmatter_key_summary import summarize_unit_frontmatter_keys


def test_frontmatter_key_summary_counts_keys_and_missing_units():
    report = summarize_unit_frontmatter_keys(
        [
            {"id": "a", "content": "---\ntitle: One\ntags:\n  - x\n---\nBody"},
            {"id": "b", "content": "Body\n---\ntitle: ignored\n---"},
            {"id": "c", "content": "---\ntitle: Two\ntitle: Again\nsource.url: u\n---"},
        ]
    )

    assert report["total_units"] == 3
    assert report["units_with_frontmatter"] == 2
    assert report["units_missing_frontmatter"] == 1
    assert report["key_counts"] == [
        {"key": "source.url", "count": 1},
        {"key": "tags", "count": 1},
        {"key": "title", "count": 2},
    ]
    assert report["duplicate_key_units"] == [{"unit_id": "c", "duplicate_keys": ["title"]}]
