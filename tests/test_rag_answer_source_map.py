from __future__ import annotations

from graph.rag.answer_source_map import build_answer_source_map


def test_build_answer_source_map_splits_sentences_and_matches_tokens():
    rows = build_answer_source_map(
        "Solar storage improves grid reliability. Cooking notes are unrelated.",
        [
            {"id": "solar", "title": "Grid storage", "content": "Solar storage reliability planning"},
            {"id": "cook", "content": "Recipe cooking notes"},
        ],
    )

    assert rows == [
        {
            "sentence": "Solar storage improves grid reliability.",
            "supporting_result_ids": ["solar"],
            "matched_terms": ["grid", "reliability", "solar", "storage"],
            "confidence": "medium",
        },
        {
            "sentence": "Cooking notes are unrelated.",
            "supporting_result_ids": ["cook"],
            "matched_terms": ["cooking", "notes"],
            "confidence": "low",
        },
    ]


def test_build_answer_source_map_citation_url_and_title_matches_raise_confidence():
    rows = build_answer_source_map(
        "The Example Report says deployment rose. See https://example.test/report for details.",
        [
            {
                "id": "a",
                "title": "Example Report",
                "url": "https://example.test/report/",
                "content": "Deployment rose in 2025.",
            }
        ],
    )

    assert rows[0]["supporting_result_ids"] == ["a"]
    assert rows[0]["confidence"] == "high"
    assert rows[1]["supporting_result_ids"] == ["a"]
    assert rows[1]["confidence"] == "high"


def test_build_answer_source_map_supports_nested_results_and_none_confidence():
    rows = build_answer_source_map(
        "No matching sentence.",
        [{"unit": {"id": "unit", "metadata": {"title": "Other"}, "content": "Different content"}}],
    )

    assert rows == [
        {
            "sentence": "No matching sentence.",
            "supporting_result_ids": [],
            "matched_terms": [],
            "confidence": "none",
        }
    ]
