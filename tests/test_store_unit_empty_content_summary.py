from __future__ import annotations

from graph.store.unit_empty_content_summary import summarize_unit_empty_content


def test_unit_empty_content_summary_classifies_empty_content_categories():
    summary = summarize_unit_empty_content(
        [
            {"id": "b", "content": ""},
            {"id": "a"},
            {"id": "c", "content": " \n\t"},
            {"id": "d", "content": "---\ntitle: Only metadata\n---"},
            {"id": "e", "content": "Body"},
        ]
    )

    assert summary == {
        "total_units": 5,
        "empty_content_units": 2,
        "whitespace_only_units": 1,
        "metadata_only_units": 1,
        "non_empty_units": 1,
        "examples": {
            "empty_content_unit_ids": ["a", "b"],
            "whitespace_only_unit_ids": ["c"],
            "metadata_only_unit_ids": ["d"],
        },
    }
