from __future__ import annotations

from graph.rag import detect_query_comparator_intent


def test_comparator_intent_detects_comparison_and_entities():
    result = detect_query_comparator_intent("Compare Postgres vs MySQL for analytics workloads.")

    assert result["has_comparator_intent"] is True
    assert result["intent_labels"] == ["compare"]
    assert result["compared_entities"] == ["Postgres", "MySQL for analytics workloads"]
    assert result["confidence"] == 0.8


def test_comparator_intent_detects_ranking_and_tradeoffs():
    result = detect_query_comparator_intent("Rank the top vendors and explain tradeoffs.")

    assert result["intent_labels"] == ["rank", "tradeoffs"]
    assert [row["intent"] for row in result["matched_terms"]] == ["rank", "tradeoffs"]


def test_non_comparative_factual_query_returns_low_confidence():
    result = detect_query_comparator_intent("What is vector search?")

    assert result["has_comparator_intent"] is False
    assert result["confidence"] == 0.0
    assert result["intent_labels"] == []
