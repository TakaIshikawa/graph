from __future__ import annotations

from graph.store.unit_frontmatter_presence_summary import summarize_unit_frontmatter_presence


def test_frontmatter_presence_summary_counts_statuses_groups_and_keys():
    report = summarize_unit_frontmatter_presence(
        [
            {"content": "---\ntitle: One\ntags:\n - x\n---\nBody", "source": "docs", "entity_type": "note"},
            {"content": "---\n---\nEmpty", "source": "docs", "entity_type": "note"},
            {"content": "---\nnot yaml\n---", "source": "web", "entity_type": "page"},
            {"content": "Body only", "source": "web", "entity_type": "page"},
        ]
    )

    assert report["valid_frontmatter_units"] == 1
    assert report["empty_frontmatter_units"] == 1
    assert report["malformed_frontmatter_units"] == 1
    assert report["missing_frontmatter_units"] == 1
    assert report["top_frontmatter_keys"] == [{"key": "tags", "count": 1}, {"key": "title", "count": 1}]
    assert {"name": "docs", "valid": 1, "empty": 1, "malformed": 0, "missing": 0} in report["by_source"]
