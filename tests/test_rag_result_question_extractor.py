from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag.result_question_extractor import extract_result_questions


@dataclass
class Result:
    id: str
    snippet: str = ""
    metadata: dict | None = None


def test_extract_result_questions_finds_questions_and_classifies_leading_words():
    rows = extract_result_questions(
        [
            {
                "id": "a",
                "content": "Why did retrieval drift? Notes continue. How should evidence be ranked?",
            },
            {"id": "b", "text": "Is the source current? What changed in 2024?"},
        ]
    )

    assert rows == [
        {"result_id": "a", "question": "Why did retrieval drift?", "question_type": "why", "position": 0},
        {
            "result_id": "a",
            "question": "How should evidence be ranked?",
            "question_type": "how",
            "position": 1,
        },
        {"result_id": "b", "question": "Is the source current?", "question_type": "other", "position": 2},
        {"result_id": "b", "question": "What changed in 2024?", "question_type": "what", "position": 3},
    ]


def test_extract_result_questions_supports_objects_tuples_and_metadata_fallbacks():
    rows = extract_result_questions(
        [
            (Result("object", snippet="When did the object update?"), 0.8),
            {"metadata": {"source_id": "meta", "content": "Where is the metadata question?"}},
        ]
    )

    assert rows == [
        {"result_id": "object", "question": "When did the object update?", "question_type": "when", "position": 0},
        {
            "result_id": "meta",
            "question": "Where is the metadata question?",
            "question_type": "where",
            "position": 1,
        },
    ]


def test_extract_result_questions_normalizes_whitespace_and_preserves_order():
    rows = extract_result_questions(
        [
            {
                "unit_id": "u1",
                "snippet": "Who owns\n\nthis item? Why now?",
            }
        ]
    )

    assert rows == [
        {"result_id": "u1", "question": "Who owns this item?", "question_type": "who", "position": 0},
        {"result_id": "u1", "question": "Why now?", "question_type": "why", "position": 1},
    ]


def test_extract_result_questions_respects_limit_and_empty_results():
    assert extract_result_questions([{"content": "Why one? How two?"}], max_questions=1) == [
        {"result_id": "result-1", "question": "Why one?", "question_type": "why", "position": 0}
    ]
    assert extract_result_questions([{"content": "No explicit question."}]) == []
    assert extract_result_questions([{"content": "Why?"}], max_questions=0) == []


@pytest.mark.parametrize("max_questions", [-1, 1.5, True, "2"])
def test_extract_result_questions_validates_max_questions(max_questions):
    with pytest.raises(ValueError, match="max_questions"):
        extract_result_questions([], max_questions=max_questions)
