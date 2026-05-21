from __future__ import annotations

from graph.rag.context_quote_density import analyze_context_quote_density


def test_context_quote_density_counts_strings_and_blockquotes():
    report = analyze_context_quote_density(['Intro "quoted span" text.', "> cited block\nplain"])

    assert report["chunk_count"] == 2
    assert report["chunks"][0]["quoted_span_count"] == 1
    assert report["chunks"][1]["blockquote_count"] == 1
    assert report["quoted_length"] > 0


def test_context_quote_density_warns_for_low_and_high_density():
    low = analyze_context_quote_density("plain context without source excerpts")
    high = analyze_context_quote_density('"all quoted"')

    assert low["warnings"] == ["under_quoted_context"]
    assert high["warnings"] == ["over_quoted_context"]
