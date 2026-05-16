from __future__ import annotations

import pytest

from graph.rag import decompose_query_for_retrieval


def test_decompose_query_for_retrieval_returns_broad_subquery_for_simple_query():
    result = decompose_query_for_retrieval("graph embeddings")

    assert result == {
        "original_query": "graph embeddings",
        "subqueries": [
            {
                "text": "graph embeddings",
                "intent": "exploratory",
                "required_terms": ["graph", "embeddings"],
                "optional_terms": [],
                "rationale": "single focused retrieval query",
            }
        ],
        "detected_constraints": {},
        "strategy": {
            "mode": "single_clause",
            "subquery_count": 1,
            "split_on": [],
            "max_subqueries": 5,
        },
    }


def test_decompose_query_for_retrieval_splits_compound_query_and_preserves_quotes():
    result = decompose_query_for_retrieval(
        'Compare "hybrid search" and vector databases after 2024 from Readwise'
    )

    assert [item["text"] for item in result["subqueries"]] == [
        'Compare "hybrid search"',
        "vector databases after 2024 from Readwise",
    ]
    assert result["subqueries"][0]["intent"] == "comparison"
    assert result["subqueries"][0]["required_terms"] == ["hybrid search", "compare"]
    assert result["subqueries"][1]["required_terms"] == ["vector", "databases", "2024", "readwise"]
    assert result["detected_constraints"] == {
        "quoted_phrases": ["hybrid search"],
        "date": {"years": ["2024"], "after": "2024"},
        "sources": ["readwise"],
        "entity_terms": ["Compare", "hybrid search", "Readwise"],
    }
    assert result["strategy"]["mode"] == "decomposed"
    assert result["strategy"]["split_on"] == ["conjunction", "comparison"]


def test_decompose_query_for_retrieval_respects_max_subqueries():
    result = decompose_query_for_retrieval("alpha and beta and gamma", max_subqueries=2)

    assert [item["text"] for item in result["subqueries"]] == ["alpha", "beta"]
    assert result["strategy"]["max_subqueries"] == 2


def test_decompose_query_for_retrieval_extracts_date_constraints():
    result = decompose_query_for_retrieval("timeline of incidents since 2021 before 2024")

    assert result["detected_constraints"]["date"] == {
        "years": ["2021", "2024"],
        "since": "2021",
        "before": "2024",
    }


def test_decompose_query_for_retrieval_validates_max_subqueries():
    with pytest.raises(ValueError, match="max_subqueries"):
        decompose_query_for_retrieval("query", max_subqueries=0)
    with pytest.raises(ValueError, match="max_subqueries"):
        decompose_query_for_retrieval("query", max_subqueries=True)
