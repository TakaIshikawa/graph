from __future__ import annotations

from types import SimpleNamespace

from graph.rag.query_constraints import extract_query_constraints


def test_query_constraints_extracts_mixed_constraints():
    report = extract_query_constraints('"exact phrase" +must -exclude site:example.com domain:docs.example.com filetype:pdf after:2024 security 2025-01-02')

    assert report["quoted_phrases"] == ["exact phrase"]
    assert report["required_terms"] == ["must"]
    assert report["excluded_terms"] == ["exclude"]
    assert report["site_filters"] == ["example.com"]
    assert report["domain_filters"] == ["docs.example.com"]
    assert report["filetypes"] == ["pdf"]
    assert "after:2024" in report["date_constraints"]
    assert "2025-01-02" in report["date_constraints"]
    assert report["has_constraints"] is True


def test_query_constraints_handles_empty_and_object_payloads():
    assert extract_query_constraints("")["has_constraints"] is False
    assert extract_query_constraints(({"query": "+required site:example.org"}, 0.7))["required_terms"] == ["required"]
    assert extract_query_constraints(SimpleNamespace(text='find "quoted item"'))["quoted_phrases"] == ["quoted item"]


def test_query_constraints_no_constraint_query_is_stable():
    report = extract_query_constraints("explain retrieval ranking basics")

    assert report["query"] == "explain retrieval ranking basics"
    assert report["quoted_phrases"] == []
    assert report["required_terms"] == []
    assert report["excluded_terms"] == []
    assert report["has_constraints"] is False
