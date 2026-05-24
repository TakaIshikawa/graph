from __future__ import annotations

from graph.rag.query_risk_tolerance_requirement import detect_query_risk_tolerance_requirement


def test_detects_conservative_risk_preference():
    result = detect_query_risk_tolerance_requirement("Recommend a cautious low-risk migration plan.")

    assert result["requires_risk_tolerance"] is True
    assert result["risk_tolerances"] == ["conservative"]
    assert set(result["matched_phrases"]) == {"cautious", "low-risk"}


def test_detects_aggressive_risk_preference():
    result = detect_query_risk_tolerance_requirement("Give me an aggressive experimental launch option.")

    assert result["risk_tolerances"] == ["aggressive"]
    assert set(result["matched_phrases"]) == {"aggressive", "experimental"}


def test_flags_safety_critical_language_separately():
    result = detect_query_risk_tolerance_requirement("Use a conservative approach for patient safety.")

    assert result["risk_tolerances"] == ["conservative"]
    assert result["safety_critical"] is True
    assert result["safety_critical_phrases"] == ["patient safety"]


def test_generic_uncertainty_is_not_risk_tolerance():
    result = detect_query_risk_tolerance_requirement("What uncertainties affect this forecast?")

    assert result["requires_risk_tolerance"] is False
    assert result["risk_tolerances"] == []
    assert result["safety_critical"] is False
