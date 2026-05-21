from __future__ import annotations

from graph.rag.query_citation_requirements import plan_query_citation_requirements


def test_high_stakes_queries_require_authoritative_recent_citations():
    report = plan_query_citation_requirements("medical dose advice for a tax-deductible treatment", result_count=2)

    assert "high_stakes" in report["intent_flags"]
    assert report["minimum_citations"] == 3
    assert report["citation_density"] == "high"
    assert report["required_citation_types"] == ["authoritative", "recent"]
    assert "result_count_below_minimum_citations" in report["warnings"]


def test_comparison_and_timeline_add_specific_requirements():
    report = plan_query_citation_requirements("Compare product history over time", result_count=4)

    assert set(report["intent_flags"]) == {"comparison", "timeline"}
    assert set(report["required_citation_types"]) == {"chronological", "comparative", "dated", "source_per_option"}


def test_short_or_empty_queries_return_warnings():
    assert plan_query_citation_requirements("", result_count=0)["warnings"] == ["empty_query"]
    assert "short_query" in plan_query_citation_requirements("tax", result_count=1)["warnings"]
