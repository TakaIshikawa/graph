from __future__ import annotations

from graph.rag.query_cost_requirement_detector import detect_query_cost_requirements


def test_query_cost_requirement_detects_low_cost_cues_and_limits():
    report = detect_query_cost_requirements("Keep it cheap, use minimal tokens, and limit to 5 sources under 2,000 tokens.")

    assert report["cost_sensitive"] is True
    assert report["budget_level"] == "low"
    assert [cue["family"] for cue in report["matched_cues"]] == ["low_cost", "low_cost"]
    assert report["requested_limits"] == [
        {"amount": 5, "unit": "sources", "text": "limit to 5 sources"},
        {"amount": 2000, "unit": "tokens", "text": "under 2,000 tokens"},
    ]


def test_query_cost_requirement_detects_high_coverage_cues():
    report = detect_query_cost_requirements("Do comprehensive deep research with maximum coverage.")

    assert report["budget_level"] == "high_coverage"
    assert report["cost_sensitive"] is False
    assert report["confidence"] == 0.8


def test_query_cost_requirement_returns_defaults_without_cues():
    report = detect_query_cost_requirements("Find sources about graph databases.")

    assert report["budget_level"] == "unspecified"
    assert report["requested_limits"] == []
