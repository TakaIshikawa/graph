from __future__ import annotations

from graph.rag import detect_query_citation_requirement


def test_query_citation_requirement_detects_citation_quote_and_primary_source_intents():
    report = detect_query_citation_requirement("Use primary sources, cite sources, and quote evidence.")

    assert report["requires_citations"] is True
    assert report["requires_quotes"] is True
    assert report["requires_primary_sources"] is True
    assert report["excludes_citations"] is False
    assert report["matched_terms"] == ["cite sources", "quote evidence", "primary sources"]


def test_query_citation_requirement_respects_no_citation_requests():
    report = detect_query_citation_requirement("No citations or links, just summarize.")

    assert report["requires_citations"] is False
    assert report["excludes_citations"] is True
