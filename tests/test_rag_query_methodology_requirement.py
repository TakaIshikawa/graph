from __future__ import annotations

from graph.rag.query_methodology_requirement import detect_query_methodology_requirements


def test_query_methodology_requirement_detects_multiple_methods():
    report = detect_query_methodology_requirements("Prioritize RCTs, cohort studies, surveys, and benchmarks.")

    assert report["has_methodology_requirement"] is True
    assert report["methodology_categories"] == ["randomized_trial", "cohort_study", "survey", "benchmark"]
    assert report["requirements"][0]["matched_spans"][0]["text"] == "RCTs"


def test_query_methodology_requirement_maps_common_phrases():
    report = detect_query_methodology_requirements("Use systematic reviews, case studies, and independent audits.")

    assert report["methodology_categories"] == ["case_study", "audit", "meta_analysis"]


def test_query_methodology_requirement_ignores_generic_words():
    report = detect_query_methodology_requirements("Review the product case and audit log if useful.")

    assert report["has_methodology_requirement"] is False
    assert report["requirements"] == []
