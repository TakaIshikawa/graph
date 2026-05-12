from __future__ import annotations

import pytest

from graph.rag.answer_gap_questions import build_answer_gap_questions


def test_answer_gap_questions_detect_missing_terms_and_common_gaps():
    rows = build_answer_gap_questions(
        "battery policy timeline",
        [{"id": "a", "title": "Battery update", "source_project": "notes"}],
    )

    assert [row["reason"] for row in rows] == [
        "low_evidence_count",
        "missing_query_term",
        "missing_query_term",
        "missing_dates",
        "single_source_result_set",
    ]
    assert rows[1]["question"] == "Which evidence specifically addresses policy?"
    assert rows[2]["question"] == "Which evidence specifically addresses timeline?"


def test_answer_gap_questions_avoids_source_and_date_gaps_when_covered():
    rows = build_answer_gap_questions(
        "battery policy",
        [
            {"title": "battery policy", "source_project": "a", "date": "2024-01-01"},
            {"summary": "policy details", "source_project": "b"},
            {"content": "battery", "source_project": "b"},
        ],
    )

    assert rows == []


def test_answer_gap_questions_respects_max_questions():
    rows = build_answer_gap_questions("alpha beta gamma delta", [], max_questions=2)

    assert len(rows) == 2
    assert [row["reason"] for row in rows] == ["low_evidence_count", "missing_query_term"]


@pytest.mark.parametrize("max_questions", [-1, 1.2, True, "2"])
def test_answer_gap_questions_validates_max_questions(max_questions):
    with pytest.raises(ValueError, match="max_questions must be a non-negative integer"):
        build_answer_gap_questions("query", [], max_questions=max_questions)
