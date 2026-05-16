from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag.evidence_sentence_ranker import rank_evidence_sentences


@dataclass
class Result:
    id: str
    content: str = ""
    metadata: dict | None = None


def test_rank_evidence_sentences_orders_by_coverage_score_result_and_position():
    rows = rank_evidence_sentences(
        [
            {
                "id": "first",
                "content": "Solar storage supports the grid. Solar storage grid.",
            },
            {"id": "second", "content": "Solar storage grid."},
        ],
        "solar storage grid",
    )

    assert [(row["result_id"], row["sentence"], row["matched_terms"]) for row in rows] == [
        ("first", "Solar storage grid.", ["solar", "storage", "grid"]),
        ("second", "Solar storage grid.", ["solar", "storage", "grid"]),
        ("first", "Solar storage supports the grid.", ["solar", "storage", "grid"]),
    ]
    assert rows[0]["score"] > rows[2]["score"]
    assert [row["position"] for row in rows] == [1, 0, 0]


def test_rank_evidence_sentences_supports_objects_tuples_and_metadata_fallbacks():
    rows = rank_evidence_sentences(
        [
            (Result("object", content="Object text mentions alpha."), 0.9),
            {"metadata": {"unit_id": "meta", "snippet": "Metadata snippet mentions beta and alpha."}},
            {"source_id": "source", "text": "Source text mentions gamma."},
        ],
        "alpha beta",
    )

    assert rows == [
        {
            "result_id": "meta",
            "sentence": "Metadata snippet mentions beta and alpha.",
            "matched_terms": ["alpha", "beta"],
            "score": 2.333333,
            "position": 0,
        },
        {
            "result_id": "object",
            "sentence": "Object text mentions alpha.",
            "matched_terms": ["alpha"],
            "score": 1.25,
            "position": 0,
        },
    ]


def test_rank_evidence_sentences_respects_limit_and_query_term_order():
    rows = rank_evidence_sentences(
        [
            {"id": "a", "content": "Beta alpha. Alpha only. Beta only."},
            {"id": "b", "content": "Alpha beta."},
        ],
        "alpha beta alpha",
        max_sentences=2,
    )

    assert rows == [
        {
            "result_id": "a",
            "sentence": "Beta alpha.",
            "matched_terms": ["alpha", "beta"],
            "score": 3.0,
            "position": 0,
        },
        {
            "result_id": "b",
            "sentence": "Alpha beta.",
            "matched_terms": ["alpha", "beta"],
            "score": 3.0,
            "position": 0,
        },
    ]


def test_rank_evidence_sentences_empty_query_no_matches_and_zero_limit_return_empty_list():
    assert rank_evidence_sentences([{"content": "alpha"}], "") == []
    assert rank_evidence_sentences([{"content": "alpha"}], "beta") == []
    assert rank_evidence_sentences([{"content": "alpha"}], "alpha", max_sentences=0) == []


@pytest.mark.parametrize("max_sentences", [-1, 1.5, True, "3"])
def test_rank_evidence_sentences_validates_max_sentences(max_sentences):
    with pytest.raises(ValueError, match="max_sentences"):
        rank_evidence_sentences([], "alpha", max_sentences=max_sentences)
