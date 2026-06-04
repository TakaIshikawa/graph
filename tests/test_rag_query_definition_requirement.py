from __future__ import annotations

from graph.rag import detect_query_definition_requirement


def test_definition_requirement_detects_definition_distinction_example_and_taxonomy():
    result = detect_query_definition_requirement(
        "What is ABAC? Explain the difference between ABAC and RBAC with examples and types of policies."
    )

    assert result["has_definition_requirement"] is True
    assert result["requirement_labels"] == ["compare_terms", "definition", "example_request", "taxonomy_request"]
    assert result["confidence"] == 0.95


def test_definition_requirement_is_case_insensitive_and_punctuation_tolerant():
    result = detect_query_definition_requirement("DEFINE: zero trust; give an example.")

    assert result["requirement_labels"] == ["definition", "example_request"]
    assert [row["matched_text"].casefold() for row in result["matched_cues"]] == ["define", "example"]


def test_definition_requirement_returns_low_confidence_for_unrelated_queries():
    result = detect_query_definition_requirement("List SOC 2 audit evidence sources.")

    assert result["has_definition_requirement"] is False
    assert result["requirement_labels"] == []
    assert result["confidence"] == 0.0
