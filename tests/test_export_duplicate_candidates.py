from __future__ import annotations

import pytest

from graph.export import export_duplicate_candidates_markdown


def test_export_duplicate_candidates_markdown_sorts_then_filters_and_limits():
    candidates = [
        {
            "unit_ids": ["unit-c", "unit-d"],
            "score": 0.91,
            "reasons": ["content"],
        },
        {
            "unit_ids": ["unit-b", "unit-a"],
            "score": 0.97,
            "reasons": ["source_id"],
        },
        {
            "unit_ids": ["unit-a", "unit-b"],
            "score": 0.97,
            "reasons": ["url"],
        },
        {
            "unit_ids": ["unit-e", "unit-f"],
            "score": 0.88,
            "reasons": ["title"],
        },
    ]

    report = export_duplicate_candidates_markdown(candidates, min_score=0.9, limit=2)

    assert "| Candidates | 2 |" in report
    assert "| Minimum score | 0.9 |" in report
    assert "| Limit | 2 |" in report
    first = report.index("| 0.97 | unit-a |")
    second = report.index("| 0.97 | unit-b |")
    assert first < second
    assert "unit-c" not in report
    assert "unit-e" not in report


def test_export_duplicate_candidates_markdown_renders_unit_titles_sources_and_reasons():
    report = export_duplicate_candidates_markdown(
        [
            {
                "units": [
                    {
                        "id": "left|id",
                        "title": "Left *title*",
                        "source_project": "max",
                        "source_entity_type": "note",
                        "source_id": "left-source",
                    },
                    {
                        "id": "right`id",
                        "title": "Right [title]",
                        "source_project": "browser",
                        "source_id": "right-source",
                    },
                ],
                "score": 1.0,
                "reasons": ["canonical|url", "title_similarity"],
                "matching_fields": {"url": "https://example.com/a|b", "title_similarity": 0.96},
            }
        ]
    )

    assert "left\\|id" in report
    assert "Left \\*title\\*" in report
    assert "max / note / left-source" in report
    assert "right\\`id" in report
    assert "Right \\[title\\]" in report
    assert "canonical\\|url; title\\_similarity" in report
    assert "https://example.com/a\\|b" in report


def test_export_duplicate_candidates_markdown_empty_results_are_valid_report():
    report = export_duplicate_candidates_markdown(
        [{"unit_ids": ["unit-a", "unit-b"], "score": 0.5, "reasons": ["title"]}],
        min_score=0.9,
    )

    assert report.startswith("# Duplicate Candidates\n")
    assert "| Candidates | 0 |" in report
    assert "_No matching duplicate candidates._" in report


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_score": "0.9"}, "min_score"),
        ({"min_score": True}, "min_score"),
        ({"limit": -1}, "limit"),
        ({"limit": 1.5}, "limit"),
        ({"limit": False}, "limit"),
    ],
)
def test_export_duplicate_candidates_markdown_validates_filters(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        export_duplicate_candidates_markdown([], **kwargs)
