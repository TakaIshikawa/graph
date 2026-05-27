from __future__ import annotations

from graph.rag.result_stakeholder_coverage import analyze_result_stakeholder_coverage


def test_detects_default_stakeholders_from_content_and_metadata():
    result = analyze_result_stakeholder_coverage([{"id": "a", "content": "Customers and admins benefit.", "metadata": {"audience": "regulator"}}])

    assert result["covered"] == ["admin", "customer", "regulator"]
    assert "vendor" in result["missing"]
    assert result["per_result_matches"][0]["matches"] == ["customer", "admin", "regulator"]


def test_custom_stakeholders_override_defaults():
    result = analyze_result_stakeholder_coverage([{"content": "Clinicians need training."}], ["clinician"])

    assert result["covered"] == ["clinician"]
    assert result["missing"] == []


def test_missing_stakeholders_are_deterministic():
    result = analyze_result_stakeholder_coverage([], ["beta", "alpha"])

    assert result["missing"] == ["beta", "alpha"]
