from __future__ import annotations

from graph.store.metadata_value_type_summary import summarize_metadata_value_types


def test_metadata_value_type_summary_distinguishes_types_and_sources():
    summary = summarize_metadata_value_types(
        [
            {"source_project": "docs", "metadata": {"flag": True, "score": 3, "tags": ["a"], "title": "Spec"}},
            {"source_project": "crm", "metadata": {"flag": None, "score": 4.5, "tags": [], "extra": {"x": 1}}},
            {"source_project": "docs", "metadata": {"title": "", "flag": False}},
        ]
    )

    assert summary["rows"] == [
        {"metadata_key": "extra", "value_type": "mapping", "unit_count": 1, "non_empty_count": 1, "example_values": ["{'x': 1}"], "sources": ["crm"]},
        {"metadata_key": "flag", "value_type": "boolean", "unit_count": 2, "non_empty_count": 2, "example_values": ["True", "False"], "sources": ["docs"]},
        {"metadata_key": "flag", "value_type": "null", "unit_count": 1, "non_empty_count": 0, "example_values": [], "sources": ["crm"]},
        {"metadata_key": "score", "value_type": "number", "unit_count": 2, "non_empty_count": 2, "example_values": ["3", "4.5"], "sources": ["crm", "docs"]},
        {"metadata_key": "tags", "value_type": "list", "unit_count": 2, "non_empty_count": 1, "example_values": ["['a']"], "sources": ["crm", "docs"]},
        {"metadata_key": "title", "value_type": "string", "unit_count": 2, "non_empty_count": 1, "example_values": ["'Spec'"], "sources": ["docs"]},
    ]


def test_metadata_value_type_summary_caps_examples_and_ignores_missing_metadata():
    summary = summarize_metadata_value_types(
        [
            {"metadata": {"k": "a"}},
            {"metadata": {"k": "b"}},
            {"metadata": {"k": "c"}},
            {"metadata": {"k": "d"}},
            {"metadata": {}},
            {},
        ],
        example_limit=2,
    )

    assert summary == {
        "rows": [
            {"metadata_key": "k", "value_type": "string", "unit_count": 4, "non_empty_count": 4, "example_values": ["'a'", "'b'"], "sources": []}
        ],
        "row_count": 1,
        "unit_count": 4,
    }
