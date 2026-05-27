from __future__ import annotations

from graph.store.unit_markdown_horizontal_rule_summary import summarize_unit_markdown_horizontal_rules


def test_horizontal_rule_summary_groups_markers_and_ignores_frontmatter_and_fences():
    summary = summarize_unit_markdown_horizontal_rules([
        {"id": "u1", "source": "s", "content": "---\ntitle: A\n---\n\n***\n_ _ _\n```\n---\n```"},
        {"id": "u2", "source": "s", "content": "- - -"},
    ])

    assert summary["sources"] == [
        {"source": "s", "unit_count": 2, "units_with_horizontal_rules": 2, "horizontal_rule_count": 3, "most_common_rule_marker": "-", "max_rules_per_unit": 2}
    ]
