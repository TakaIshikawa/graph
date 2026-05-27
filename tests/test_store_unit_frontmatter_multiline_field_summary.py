from __future__ import annotations

from graph.store import summarize_unit_frontmatter_multiline_fields


def test_frontmatter_multiline_summary_detects_literal_and_folded_fields():
    report = summarize_unit_frontmatter_multiline_fields([{"id": "a", "content": "---\nsummary: |-\n  hi\nnotes: >+\n  there\n---\nbody"}])

    assert report["total_multiline_fields"] == 2
    assert report["style_counts"] == {"folded": 1, "literal": 1}
    assert report["chomping_counts"] == {"keep": 1, "strip": 1}


def test_frontmatter_multiline_summary_scans_only_leading_frontmatter():
    report = summarize_unit_frontmatter_multiline_fields([{"content": "body\n---\nsummary: |\n---"}])

    assert report["units_with_frontmatter"] == 0


def test_frontmatter_multiline_summary_reports_key_counts_and_examples():
    report = summarize_unit_frontmatter_multiline_fields([{"id": "a", "content": "---\ndesc: |\n  x\n---"}])

    assert report["key_counts"] == {"desc": 1}
    assert report["examples"] == [{"unit_id": "a", "key": "desc", "style": "literal", "chomping": "clip", "line": 2}]
