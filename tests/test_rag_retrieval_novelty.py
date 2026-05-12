from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag.retrieval_novelty import score_retrieval_novelty


@dataclass
class Result:
    id: str
    text: str
    source_project: str


def test_retrieval_novelty_scores_incremental_overlap_and_duplicates():
    rows = score_retrieval_novelty(
        [
            {"id": "a", "title": "Battery storage grid", "source_project": "notes"},
            {"id": "b", "content": "Battery storage grid update", "source_project": "notes"},
            Result("c", "Solar market outlook", "web"),
        ],
        similarity_threshold=0.65,
    )

    assert rows == [
        {
            "id": "a",
            "source_project": "notes",
            "novelty_score": 1,
            "duplicate_of": None,
            "shared_terms": [],
            "token_count": 3,
        },
        {
            "id": "b",
            "source_project": "notes",
            "novelty_score": 0,
            "duplicate_of": "a",
            "shared_terms": ["battery", "grid", "storage"],
            "token_count": 4,
        },
        {
            "id": "c",
            "source_project": "web",
            "novelty_score": 1,
            "duplicate_of": None,
            "shared_terms": [],
            "token_count": 3,
        },
    ]


def test_metadata_text_fields_and_ordered_ties_are_stable():
    rows = score_retrieval_novelty(
        [
            {"id": "a", "metadata": {"summary": "alpha beta"}},
            {"id": "b", "summary": "alpha beta"},
            {"id": "c", "text": "alpha beta"},
        ],
        similarity_threshold=1,
    )

    assert [row["duplicate_of"] for row in rows] == [None, "a", "a"]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True, "bad"])
def test_retrieval_novelty_validates_threshold(threshold):
    with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
        score_retrieval_novelty([], similarity_threshold=threshold)
