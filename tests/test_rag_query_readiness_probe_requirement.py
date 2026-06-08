from __future__ import annotations

from graph.rag.query_readiness_probe_requirement import detect_query_readiness_probe_requirement


def test_detects_readiness_probe_categories():
    result = detect_query_readiness_probe_requirement(
        "For a Kubernetes service, require readiness probe, /readyz endpoint, gate traffic, "
        "dependency readiness, mark ready after startup, and readiness timeout."
    )

    assert result["has_readiness_probe_requirement"] is True
    assert result["requirements"] == [
        {"category": "dependency_readiness", "matched_text": "dependency readiness", "severity": "high"},
        {"category": "failure_threshold", "matched_text": "readiness timeout", "severity": "medium"},
        {"category": "readiness_probe", "matched_text": "readiness probe", "severity": "high"},
        {"category": "ready_endpoint", "matched_text": "/readyz", "severity": "medium"},
        {"category": "startup_readiness", "matched_text": "mark ready after startup", "severity": "medium"},
        {"category": "traffic_gating", "matched_text": "gate traffic", "severity": "high"},
    ]


def test_detects_service_removal_and_dependency_ready_wording():
    result = detect_query_readiness_probe_requirement(
        "Application health should remove from service until dependencies are ready."
    )

    assert [row["category"] for row in result["requirements"]] == ["dependency_readiness", "traffic_gating"]
    assert all(row["severity"] == "high" for row in result["requirements"])


def test_ignores_general_project_readiness_text():
    assert detect_query_readiness_probe_requirement("Assess team readiness for a launch presentation.") == {
        "has_readiness_probe_requirement": False,
        "requirements": [],
    }
