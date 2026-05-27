from __future__ import annotations

from graph.store import summarize_unit_yaml_alias_anchors


def test_yaml_alias_anchor_summary_frontmatter_only():
    report = summarize_unit_yaml_alias_anchors([{"id": "u", "content": "---\na: &main 1\nb: *main\nc: *missing\nd: &main 2\n---\nprose &ignored *ignored"}])

    assert report["anchor_count"] == 2
    assert report["alias_count"] == 2
    assert report["reused_anchor_names"] == [{"name": "main", "count": 2}]
    assert report["unresolved_aliases"] == [{"name": "missing", "count": 1}]
