from __future__ import annotations

from graph.store.unit_yaml_date_field_summary import summarize_unit_yaml_date_fields


def test_unit_yaml_date_fields_classifies_default_fields():
    summary = summarize_unit_yaml_date_fields(
        [
            {"id": "u1", "content": "---\ncreated: 2024-01-01\nupdated: bad\n---\nBody"},
            {"id": "u2", "content": "---\ndate: 2024-02-30\npublished: 2024-03-01\n---"},
        ]
    )

    assert summary["field_counts"]["created"] == {"valid": 1, "missing": 1, "invalid": 0}
    assert summary["field_counts"]["updated"] == {"valid": 0, "missing": 1, "invalid": 1}
    assert summary["invalid_examples"] == [
        {"unit_id": "u1", "field": "updated", "value": "bad"},
        {"unit_id": "u2", "field": "date", "value": "2024-02-30"},
    ]


def test_unit_yaml_date_fields_accepts_custom_fields_and_missing_frontmatter():
    summary = summarize_unit_yaml_date_fields([{"id": "u1", "content": "Body"}], field_names=["due"])

    assert summary == {"total_units": 1, "field_counts": {"due": {"valid": 0, "missing": 1, "invalid": 0}}, "invalid_examples": []}
