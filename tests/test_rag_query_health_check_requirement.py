from __future__ import annotations

from graph.rag.query_health_check_requirement import detect_query_health_check_requirement


def test_detects_kubernetes_probe_and_endpoint_requirements():
    result = detect_query_health_check_requirement(
        "For a Kubernetes service, define liveness probe, readiness probe, startup probe, /healthz, "
        "dependency health, and heartbeat behavior."
    )

    assert result["has_health_check_requirement"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "dependency_health",
        "health_endpoint",
        "heartbeat",
        "liveness_probe",
        "readiness_probe",
        "startup_probe",
    ]


def test_detects_generic_service_health_wording():
    result = detect_query_health_check_requirement(
        "Need an HTTP health endpoint for the service plus dependency checks and keep-alive signals."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "dependency_health",
        "health_endpoint",
        "heartbeat",
    ]
    assert result["requirements"][0]["severity"] == "high"


def test_ignores_general_wellness_and_medical_health_text():
    assert detect_query_health_check_requirement("Compare employee wellness benefits and medical health outcomes.") == {
        "has_health_check_requirement": False,
        "requirements": [],
    }
