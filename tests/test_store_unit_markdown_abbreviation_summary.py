from __future__ import annotations

from graph.store.unit_markdown_abbreviation_summary import summarize_unit_markdown_abbreviations


def test_summarize_unit_markdown_abbreviations_groups_expansions():
    summary = summarize_unit_markdown_abbreviations(
        [
            {"id": "u1", "content": "*[API]: Application Programming Interface\n```md\n*[API]: Ignore\n```"},
            {"id": "u2", "content": "*[API]: Active Pharmaceutical Ingredient\n*[CPU]: Central Processing Unit"},
        ]
    )

    assert summary["abbreviations"][0] == {
        "abbreviation": "API",
        "definition_count": 2,
        "unit_count": 2,
        "distinct_expansions": ["Active Pharmaceutical Ingredient", "Application Programming Interface"],
        "examples": [
            {"unit_id": "u1", "line": 1, "expansion": "Application Programming Interface"},
            {"unit_id": "u2", "line": 1, "expansion": "Active Pharmaceutical Ingredient"},
        ],
    }
    assert summary["abbreviations"][1]["abbreviation"] == "CPU"
