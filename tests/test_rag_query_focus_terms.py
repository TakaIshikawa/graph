from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag import extract_query_focus_terms


@dataclass
class Result:
    title: str
    tags: list[str]
    metadata: dict
    source_project: str
    source_id: str


def test_query_focus_terms_normalizes_punctuation_repeats_and_stopwords():
    rows = extract_query_focus_terms("What is Battery, battery storage? A x")

    assert rows == [
        {"term": "battery", "score": 2, "query_count": 2, "result_count": 0, "sources": []},
        {"term": "storage", "score": 1, "query_count": 1, "result_count": 0, "sources": []},
    ]


def test_query_focus_terms_boosts_title_tags_and_metadata_keywords():
    rows = extract_query_focus_terms(
        "battery retention policy",
        results=[
            {
                "id": "a",
                "source_project": "papers",
                "source_id": "one",
                "title": "Battery degradation",
                "tags": ["Retention"],
            },
            Result(
                title="Operations",
                tags=[],
                metadata={"keywords": [{"keyword": "Policy"}]},
                source_project="notes",
                source_id="two",
            ),
        ],
    )

    assert rows == [
        {
            "term": "battery",
            "score": 1.5,
            "query_count": 1,
            "result_count": 1,
            "sources": ["papers:one"],
        },
        {
            "term": "policy",
            "score": 1.5,
            "query_count": 1,
            "result_count": 1,
            "sources": ["notes:two"],
        },
        {
            "term": "retention",
            "score": 1.5,
            "query_count": 1,
            "result_count": 1,
            "sources": ["papers:one"],
        },
    ]


def test_query_focus_terms_applies_max_terms_after_deterministic_sort():
    rows = extract_query_focus_terms("gamma beta alpha beta", max_terms=2)

    assert [row["term"] for row in rows] == ["beta", "alpha"]


@pytest.mark.parametrize("max_terms", [0, -1, True, "3"])
def test_query_focus_terms_validates_max_terms(max_terms):
    with pytest.raises(ValueError, match="max_terms must be a positive integer"):
        extract_query_focus_terms("battery", max_terms=max_terms)
