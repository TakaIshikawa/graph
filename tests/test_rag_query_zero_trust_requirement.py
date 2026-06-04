from __future__ import annotations

from graph.rag.query_zero_trust_requirement import detect_query_zero_trust_requirements


def test_zero_trust_context_gates_specific_requirement_rows():
    result = detect_query_zero_trust_requirements("Require least privilege for admin users.")

    assert result["has_zero_trust_requirements"] is False
    assert result["requirements"] == []


def test_zero_trust_phrases_and_implementation_cues_are_detected():
    result = detect_query_zero_trust_requirements(
        "Zero trust should use explicit verification, least privilege access, device posture, "
        "microsegmentation, continuous verification, and policy enforcement."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "continuous_evaluation",
        "device_context",
        "identity_verification",
        "least_privilege",
        "network_segmentation",
        "policy_enforcement",
    ]
    assert all(row["matched_text"] and row["severity"] for row in result["requirements"])


def test_unrelated_access_control_query_returns_false():
    result = detect_query_zero_trust_requirements("How should RBAC roles map to billing permissions?")

    assert result["has_zero_trust_requirements"] is False
    assert result["requirements"] == []
