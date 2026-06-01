from __future__ import annotations

from graph.store import summarize_unit_markdown_heading_anchor_collisions


def test_heading_anchor_collision_summary_reports_only_units_with_duplicate_generated_anchors():
    summary = summarize_unit_markdown_heading_anchor_collisions(
        [
            {"id": "u1", "content": "# Hello, World!\n## Hello World\n### Other\n```\n# Hello World\n```"},
            {"id": "u2", "content": "# Unique"},
        ]
    )

    assert summary == {
        "total_units": 2,
        "affected_unit_count": 1,
        "duplicate_anchor_count": 1,
        "units": [
            {
                "unit_id": "u1",
                "duplicate_anchor_count": 1,
                "collisions": [{"anchor": "hello-world", "heading_count": 2, "levels": [1, 2], "sample_headings": ["Hello, World!", "Hello World"]}],
            }
        ],
    }
