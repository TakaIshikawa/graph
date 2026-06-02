from __future__ import annotations

from graph.rag.query_dependency_scanning_requirement import detect_query_dependency_scanning_requirement


def test_detects_sca_cvss_threshold_and_remediation_sla():
    result = detect_query_dependency_scanning_requirement(
        "Require SCA, dependency vulnerability scanning, CVSS >= 7.0 threshold, and remediation SLA within 30 days."
    )

    assert result == {
        "requires_dependency_scanning": True,
        "cue_categories": ["sca", "dependency_vulnerability_scanning", "cvss_threshold", "remediation_sla"],
        "severity_thresholds": ["CVSS >= 7.0", "30 days"],
    }


def test_detects_vulnerable_package_alerts_and_severity_language():
    result = detect_query_dependency_scanning_requirement(
        "Do vulnerable package alerts cover high severity dependencies?"
    )

    assert result["cue_categories"] == ["vulnerable_package_alerts"]
    assert result["severity_thresholds"] == ["high severity"]


def test_generic_package_comparison_does_not_match():
    assert detect_query_dependency_scanning_requirement("Compare package manager download trends.") == {
        "requires_dependency_scanning": False,
        "cue_categories": [],
        "severity_thresholds": [],
    }
