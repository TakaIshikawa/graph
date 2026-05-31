from __future__ import annotations

from graph.store.unit_frontmatter_required_field_summary import summarize_unit_frontmatter_required_fields


def test_required_frontmatter_fields_report_missing_blank_and_present_values():
    summary = summarize_unit_frontmatter_required_fields(
        [
            {"id": "u1", "content": "---\ntitle: Alpha\nsource: \n---\nBody"},
            {"id": "u2", "content": "---\ntitle: Beta\nsource: Web\n---\nBody"},
            {"id": "u3", "content": "No frontmatter"},
        ],
        ["title", "source"],
    )

    assert summary["field_counts"] == [
        {"field": "title", "missing": 1, "blank": 0, "present": 2},
        {"field": "source", "missing": 1, "blank": 1, "present": 1},
    ]
    assert summary["examples"] == [
        {"unit_id": "u1", "missing_fields": [], "blank_fields": ["source"], "present_fields": ["title"]},
        {"unit_id": "u3", "missing_fields": ["title", "source"], "blank_fields": [], "present_fields": []},
    ]


def test_required_frontmatter_fields_are_case_sensitive():
    summary = summarize_unit_frontmatter_required_fields([{"id": "u1", "content": "---\nTitle: Alpha\n---"}], ["title", "Title"])

    assert summary["field_counts"] == [
        {"field": "title", "missing": 1, "blank": 0, "present": 0},
        {"field": "Title", "missing": 0, "blank": 0, "present": 1},
    ]
