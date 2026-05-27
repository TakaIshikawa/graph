from __future__ import annotations

from graph.rag.query_comparison_requirement import detect_query_comparison_requirement


def test_detects_versus_query_and_extracts_entities():
    result = detect_query_comparison_requirement("Compare Pinecone vs Weaviate for hybrid search.")

    assert result["requires_comparison"] is True
    assert result["comparison_type"] == "compare"
    assert result["entities"] == ["Pinecone", "Weaviate"]
    assert result["matched_terms"] == ["Compare", "vs"]
    assert result["confidence"] >= 0.9


def test_detects_difference_preference_tradeoff_and_alternative_language():
    cases = [
        ("What is the difference between BM25 and vector search?", "difference", ["BM25", "vector search"]),
        ("Is RAG better than fine-tuning?", "preference", ["RAG", "fine-tuning"]),
        ("Pros and cons of Postgres and Elasticsearch", "tradeoff", ["Postgres", "Elasticsearch"]),
        ("Find alternatives to LangChain or LlamaIndex", "alternatives", ["LangChain", "LlamaIndex"]),
    ]

    for query, comparison_type, entities in cases:
        result = detect_query_comparison_requirement(query)

        assert result["requires_comparison"] is True
        assert result["comparison_type"] == comparison_type
        assert result["entities"][:2] == entities
        assert result["matched_terms"]


def test_detects_a_or_b_choice_phrasing():
    result = detect_query_comparison_requirement("Should I use Redis or DynamoDB for session storage?")

    assert result["requires_comparison"] is True
    assert result["comparison_type"] == "choice"
    assert result["entities"] == ["Redis", "DynamoDB"]
    assert "Should I use" in result["matched_terms"]


def test_unrelated_query_is_not_comparison():
    result = detect_query_comparison_requirement("Explain retrieval augmented generation for a new engineer.")

    assert result == {
        "query": "Explain retrieval augmented generation for a new engineer.",
        "requires_comparison": False,
        "comparison_type": "none",
        "entities": [],
        "matched_terms": [],
        "confidence": 0.0,
    }


def test_empty_query_is_not_comparison():
    assert detect_query_comparison_requirement("   ") == {
        "query": "",
        "requires_comparison": False,
        "comparison_type": "none",
        "entities": [],
        "matched_terms": [],
        "confidence": 0.0,
    }
