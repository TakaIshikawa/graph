from __future__ import annotations

from types import SimpleNamespace

from graph.rag.result_definition_coverage import analyze_result_definition_coverage


def test_detects_defined_query_terms_from_results():
    report = analyze_result_definition_coverage("What is retrieval augmented generation?", [{"id": "r1", "text": "Retrieval augmented generation is a method for grounding generated answers."}])

    assert report["query_terms"] == ["retrieval augmented generation"]
    assert report["defined_terms"] == ["retrieval augmented generation"]
    assert report["missing_terms"] == []


def test_reports_missing_requested_definitions_with_reason_counts():
    report = analyze_result_definition_coverage("define vector database", [{"id": "r1", "text": "This result gives examples only."}])

    assert report["missing_terms"] == ["vector database"]
    assert report["reason_counts"]["missing_requested_definition"] == 1
    assert "missing_requested_definitions" in report["warnings"]


def test_handles_acronym_queries_case_insensitively_and_tuple_objects():
    result = (SimpleNamespace(id="obj", text="RAG stands for retrieval augmented generation."), 0.9)
    report = analyze_result_definition_coverage("RAG acronym", [result])

    assert report["defined_terms"] == ["RAG"]
    assert report["results"][0]["result_id"] == "obj"
